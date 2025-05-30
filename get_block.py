import os
import requests
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from websockets import connect
import asyncio
import json
import streamlit as st

# --- Config ---
ETHERSCAN_API = "972W1N6UZ2IC6MXZJ32G7JJJT4UNMRNP6B"
ALCHEMY_WS_URL = "https://eth-mainnet.g.alchemy.com/v2/RELC1tew5qdPp0NLc82Nw"
FLASHBOTS_GQL    = "https://datasets.flashbots.net/v1/graphql"
PAGE_SIZE        = 10000

# --- Helpers ---

def safe_get(url, params=None, timeout=10):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# --- Fetch Etherscan TXs ---
def get_high_gas_txs(min_gas=100, limit=1000):
    all_txs, page = [], 1
    while len(all_txs) < limit:
        params = dict(
            module='account', action='txlist',
            address='0x0000000000000000000000000000000000000000',
            startblock=0, endblock=99999999,
            page=page, offset=PAGE_SIZE,
            sort='desc', apikey=ETHERSCAN_API
        )
        data = safe_get('https://api.etherscan.io/api', params)
        if not data or data.get('status') != '1':
            break
        df = pd.DataFrame(data['result'])
        df['gasPrice'] = df['gasPrice'].astype(float) / 1e9
        df = df[df['gasPrice'] > min_gas]
        all_txs.extend(df.to_dict('records'))
        if len(data['result']) < PAGE_SIZE:
            break
        page += 1
    txs = pd.DataFrame(all_txs).head(limit)
    return (
        txs.rename(columns={'hash':'tx_hash','from':'from_address','to':'to_address'})
           .assign(
               gasPrice=txs['gasPrice'],
               value=txs['value'].astype(float)/1e18,
               blockNumber=txs['blockNumber'].astype(int)
           )
           [['tx_hash','from_address','to_address','gasPrice','value','blockNumber']]
    )

# --- Fetch Flashbots MEV via GraphQL ---
def fetch_flashbots_mev(num_blocks=10):
    query = '''
    query($n: Int!) {
      blocks(last: $n) {
        number
        miner
        timestamp
        transactions {
          transaction_hash
          coinbase_transfer
        }
      }
    }
    '''
    try:
        resp = requests.post(
            FLASHBOTS_GQL,
            json={'query': query, 'variables': {'n': num_blocks}},
            timeout=10
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Flashbots query failed: {e}")
        return pd.DataFrame()
    data = resp.json().get('data', {}).get('blocks', [])
    records = []
    for blk in data:
        for tx in blk['transactions']:
            records.append({
                'blockNumber': blk['number'],
                'miner': blk['miner'],
                'tx_hash': tx['transaction_hash'],
                'profit': int(tx['coinbase_transfer']) / 1e18
            })
    return pd.DataFrame(records)

# --- MEV Detection ---

def detect_sandwich(txs):
    recs = []
    for i in range(1, len(txs)-1):
        p, c, n = txs.iloc[i-1], txs.iloc[i], txs.iloc[i+1]
        if (p['to_address']==n['to_address'] and
            p['gasPrice']>c['gasPrice']<n['gasPrice'] and
            p['blockNumber']==c['blockNumber']==n['blockNumber']):
            recs.append({
                'front': p['tx_hash'], 'victim': c['tx_hash'], 'back': n['tx_hash'],
                'to': p['to_address'], 'block': p['blockNumber'],
                'front_gas': p['gasPrice'], 'victim_gas': c['gasPrice'], 'back_gas': n['gasPrice']
            })
    return pd.DataFrame(recs)

def detect_anomalies(txs):
    feats = txs[['gasPrice','value']].values
    Xs = StandardScaler().fit_transform(feats)
    labels = IsolationForest(contamination=0.01, random_state=42).fit_predict(Xs)
    return txs[labels==-1]

def dbscan_cluster(txs):
    feats = txs[['gasPrice','blockNumber']].values
    Xs = StandardScaler().fit_transform(feats)
    lbls = DBSCAN(eps=0.5, min_samples=3).fit_predict(Xs)
    txs['cluster'] = lbls
    return txs[txs['cluster']!=-1]

# --- Real-time mempool ---
async def listen_mempool():
    async with connect(ALCHEMY_WS_URL) as ws:
        await ws.send(json.dumps({
            'jsonrpc':'2.0','id':1,
            'method':'eth_subscribe','params':['newPendingTransactions']
        }))
        while True:
            msg = json.loads(await ws.recv())
            if 'params' in msg:
                st.write(f"New pending TX: {msg['params']['result']}")

# --- Streamlit Dashboard ---

def run_dashboard():
    st.title("MEV Bot Detector with Flashbots MEV Bundles")
    # Sidebar settings
    min_gas     = st.sidebar.number_input("Min Gas (Gwei)", 100)
    limit       = st.sidebar.number_input("TX Limit", 500)
    num_blocks  = st.sidebar.number_input("Flashbots Blocks", 10)

    # 1. High-Gas Etherscan TXs
    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return
    st.subheader("High-Gas Transactions")
    st.dataframe(txs)

    # 2. Sandwich Attacks
    sandwiches = detect_sandwich(txs)
    st.subheader(f"Sandwich Attacks: {len(sandwiches)}")
    if not sandwiches.empty:
        st.dataframe(sandwiches)
        for _, r in sandwiches.iterrows():
            st.markdown(
                f"• In block **{r.block}**, tx **{r.victim}** was sandwiched: "
                f"front-runner **{r.front}** paid {r.front_gas:.1f} Gwei, victim paid {r.victim_gas:.1f} Gwei, "
                f"back-runner **{r.back}** paid {r.back_gas:.1f} Gwei."
            )

    # 3. Anomalous Transactions
    anomalies = detect_anomalies(txs)
    st.subheader(f"Anomalous Transactions: {len(anomalies)}")
    if not anomalies.empty:
        avg_gas = txs['gasPrice'].mean()
        st.dataframe(anomalies)
        for _, a in anomalies.iterrows():
            ratio = a.gasPrice / avg_gas if avg_gas else np.nan
            st.markdown(
                f"• Tx **{a.tx_hash}** bid {a.gasPrice:.1f} Gwei (~{ratio:.1f}× avg {avg_gas:.1f}), "
                f"but only moved {a.value:.4f} ETH. That's unusual for a high-premium bid."
            )

    # 4. Flashbots MEV Bundles
    flash = fetch_flashbots_mev(num_blocks)
    st.subheader(f"Flashbots MEV Bundles (last {num_blocks} blocks)")
    if not flash.empty:
        st.dataframe(flash)

    # 5. MEV Bot Clusters
    clusters = dbscan_cluster(txs)
    if not clusters.empty:
        st.subheader("MEV Bot Clusters")
        st.vega_lite_chart(clusters, {
            'mark':'circle','encoding':{
                'x':{'field':'blockNumber','type':'quantitative'},
                'y':{'field':'gasPrice','type':'quantitative'},
                'color':{'field':'cluster','type':'nominal'}
            }
        })

if __name__ == "__main__":
    run_dashboard()
    # To enable real-time feed: asyncio.run(listen_mempool())

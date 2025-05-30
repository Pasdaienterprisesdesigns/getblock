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
FLASHBOTS_GQL  = "https://datasets.flashbots.net/v1/graphql"
PAGE_SIZE      = 10000

# --- Helpers ---

def safe_get(url, params=None, timeout=10):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        # Silently return None; errors handled upstream
        return None
    except ValueError:
        # JSON decode error
        return None

# --- Fetch Etherscan TXs ---

def get_high_gas_txs(min_gas=100, limit=1000):
    all_txs, page = [], 1
    while len(all_txs) < limit:
        data = safe_get('https://api.etherscan.io/api', {
            'module': 'account', 'action': 'txlist',
            'address': '0x0000000000000000000000000000000000000000',
            'startblock': 0, 'endblock': 99999999,
            'page': page, 'offset': PAGE_SIZE,
            'sort': 'desc', 'apikey': ETHERSCAN_API
        })
        if not data or data.get('status') != '1':
            break
        df_page = pd.DataFrame(data['result'])
        df_page['gasPrice'] = df_page['gasPrice'].astype(float) / 1e9
        df_page = df_page[df_page['gasPrice'] > min_gas]
        all_txs.extend(df_page.to_dict('records'))
        if len(data['result']) < PAGE_SIZE:
            break
        page += 1
    txs = pd.DataFrame(all_txs).head(limit)
    txs['gasPrice'] = txs['gasPrice'].astype(float)
    txs['value'] = txs['value'].astype(float) / 1e18
    txs['blockNumber'] = txs['blockNumber'].astype(int)
    txs = txs.rename(columns={'hash':'tx_hash', 'from':'from_address', 'to':'to_address'})
    return txs[['tx_hash','from_address','to_address','gasPrice','value','blockNumber']]

# --- Fetch Flashbots MEV via GraphQL ---

def fetch_flashbots_mev(num_blocks=10):
    query = '''
    query($n: Int!) {
      blocks(last: $n) {
        number
        miner
        transactions {
          transaction_hash
          coinbase_transfer
        }
      }
    }
    '''
    try:
        resp = requests.post(FLASHBOTS_GQL,
                             json={'query': query, 'variables': {'n': num_blocks}},
                             timeout=10)
        resp.raise_for_status()
        data = resp.json().get('data', {}).get('blocks', [])
    except (requests.RequestException, ValueError):
        # Hide internal errors – return empty DataFrame for display
        return pd.DataFrame()

    records = []
    for blk in data:
        for tx in blk.get('transactions', []):
            records.append({
                'blockNumber': blk.get('number'),
                'miner': blk.get('miner'),
                'tx_hash': tx.get('transaction_hash'),
                'profit': int(tx.get('coinbase_transfer', 0)) / 1e18
            })
    return pd.DataFrame(records)

# --- Detection & Clustering ---

def detect_sandwich(txs):
    recs = []
    for i in range(1, len(txs)-1):
        prev, curr, nxt = txs.iloc[i-1], txs.iloc[i], txs.iloc[i+1]
        if (prev['to_address'] == nxt['to_address'] and
            prev['gasPrice'] > curr['gasPrice'] < nxt['gasPrice'] and
            prev['blockNumber'] == curr['blockNumber'] == nxt['blockNumber']):
            recs.append({
                'front': prev['tx_hash'], 'victim': curr['tx_hash'], 'back': nxt['tx_hash'],
                'to': prev['to_address'], 'block': prev['blockNumber'],
                'front_gas': prev['gasPrice'], 'victim_gas': curr['gasPrice'], 'back_gas': nxt['gasPrice']
            })
    return pd.DataFrame(recs)

def detect_anomalies(txs):
    feats = txs[['gasPrice','value']].values
    labels = IsolationForest(contamination=0.01, random_state=42)
    preds = labels.fit_predict(StandardScaler().fit_transform(feats))
    return txs[preds == -1]

def dbscan_cluster(txs):
    feats = txs[['gasPrice','blockNumber']].values
    lbls = DBSCAN(eps=0.5, min_samples=3).fit_predict(
        StandardScaler().fit_transform(feats))
    txs['cluster'] = lbls
    return txs[txs['cluster'] != -1]

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
    st.title("MEV Bot Detector (Clean)")
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", value=100)
    limit = st.sidebar.number_input("TX Limit", value=500)
    num_blocks = st.sidebar.number_input("Flashbots Blocks", value=10)

    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return

    st.subheader("High-Gas Transactions")
    st.dataframe(txs)

    sandwiches = detect_sandwich(txs)
    st.subheader(f"Sandwich Attacks: {len(sandwiches)}")
    if not sandwiches.empty:
        st.dataframe(sandwiches)
        for _, r in sandwiches.iterrows():
            st.markdown(
                f"• Block {r.block}: tx {r.victim} was sandwich-attacked by {r.front} and {r.back}, "
                f"bidding {r.front_gas:.1f}, {r.victim_gas:.1f}, and {r.back_gas:.1f} Gwei respectively."
            )

    anomalies = detect_anomalies(txs)
    st.subheader(f"Anomalous Transactions: {len(anomalies)}")
    if not anomalies.empty:
        avg_gas = txs['gasPrice'].mean()
        st.dataframe(anomalies)
        for _, a in anomalies.iterrows():
            ratio = a.gasPrice / avg_gas if avg_gas else np.nan
            st.markdown(
                f"• Tx {a.tx_hash} bid {a.gasPrice:.1f} Gwei (~{ratio:.1f}× avg {avg_gas:.1f}), "
                f"moved {a.value:.4f} ETH. Unusual high-premium/low-value combo."
            )

    flash = fetch_flashbots_mev(num_blocks)
    st.subheader(f"Flashbots MEV Bundles (last {num_blocks} blocks)")
    if flash.empty:
        st.write("(Flashbots data unavailable)")
    else:
        st.dataframe(flash)

    clusters = dbscan_cluster(txs)
    st.subheader("MEV Bot Clusters")
    if clusters.empty:
        st.write("(No clusters to plot)")
    else:
        st.vega_lite_chart(
            clusters,
            {'mark':'circle',
             'encoding':{
                 'x':{'field':'blockNumber','type':'quantitative'},
                 'y':{'field':'gasPrice','type':'quantitative'},
                 'color':{'field':'cluster','type':'nominal'}
             }
            },
            use_container_width=True
        )

if __name__ == "__main__":
    run_dashboard()
    # asyncio.run(listen_mempool())

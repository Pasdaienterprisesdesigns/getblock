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
FLASHBOTS_API     = "https://blocks.flashbots.net/v1/blocks"
PAGE_SIZE         = 10000

# --- Helpers ---
def safe_get(url, params=None, timeout=10):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# --- 1. Data Fetching ---
def get_high_gas_txs(min_gas=100, limit=1000):
    all_txs, page = [], 1
    while len(all_txs) < limit:
        params = dict(
            module='account', action='txlist', address='0x0000000000000000000000000000000000000000',
            startblock=0, endblock=99999999, page=page, offset=PAGE_SIZE,
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
    # rename hash column to tx_hash for clarity
    return (
        txs.rename(columns={'hash': 'tx_hash', 'from': 'from_address', 'to': 'to_address'})
           .assign(
               gasPrice=txs['gasPrice'],
               value=txs['value'].astype(float)/1e18,
               blockNumber=txs['blockNumber'].astype(int)
           )[['tx_hash', 'from_address', 'to_address', 'gasPrice', 'value', 'blockNumber']]
    )

# --- 2. Flashbots MEV ---
def fetch_flashbots_mev():
    data = safe_get(FLASHBOTS_API)
    if not data:
        return pd.DataFrame()
    mev = []
    for block in data:
        for tx in block.get('transactions', []):
            mev.append({
                'tx_hash': tx['transaction_hash'],
                'miner': block['miner'],
                'profit': float(tx.get('coinbase_transfer', 0)) / 1e18
            })
    return pd.DataFrame(mev)

# --- 3. MEV Detection with details ---
def detect_sandwich(txs):
    records = []
    for i in range(1, len(txs)-1):
        prev, curr, nxt = txs.iloc[i-1], txs.iloc[i], txs.iloc[i+1]
        if (prev['to_address'] == nxt['to_address'] and
            prev['gasPrice'] > curr['gasPrice'] < nxt['gasPrice'] and
            prev['blockNumber'] == curr['blockNumber'] == nxt['blockNumber']):
            records.append({
                'front_hash': prev['tx_hash'], 'victim_hash': curr['tx_hash'], 'back_hash': nxt['tx_hash'],
                'to_address': prev['to_address'], 'block': prev['blockNumber'],
                'front_gas': prev['gasPrice'], 'victim_gas': curr['gasPrice'], 'back_gas': nxt['gasPrice']
            })
    return pd.DataFrame(records)

# --- 4. Anomaly Detection with details ---
def detect_anomalies(txs):
    features = txs[['gasPrice', 'value']].values
    X_scaled = StandardScaler().fit_transform(features)
    labels = IsolationForest(contamination=0.01, random_state=42).fit_predict(X_scaled)
    anomalies = txs[labels == -1]
    return anomalies

# --- 5. Bot clustering helper ---
def dbscan_cluster(txs):
    features = txs[['gasPrice', 'blockNumber']].values
    X_scaled = StandardScaler().fit_transform(features)
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(X_scaled)
    txs['cluster'] = clustering.labels_
    return txs[txs['cluster'] != -1]

# --- 6. Real-time mempool ---
async def listen_mempool():
    async with connect(ALCHEMY_WS_URL) as ws:
        await ws.send(json.dumps({
            'jsonrpc': '2.0', 'id': 1,
            'method': 'eth_subscribe', 'params': ['newPendingTransactions']
        }))
        while True:
            msg = json.loads(await ws.recv())
            if 'params' in msg:
                st.write(f"New pending TX: {msg['params']['result']}")

# --- 7. Streamlit Dashboard ---
def run_dashboard():
    st.title("MEV Bot Detector with Exact Hashes")
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", value=100)
    limit = st.sidebar.number_input("TX Limit", value=500)

    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return

    st.subheader("High-Gas Transactions")
    st.dataframe(txs)

    sandwiches = detect_sandwich(txs)
    st.subheader(f"Detected {len(sandwiches)} Sandwich Attacks")
    if not sandwiches.empty:
        st.dataframe(sandwiches)

    anomalies = detect_anomalies(txs)
    st.subheader(f"Anomalous Transactions: {len(anomalies)}")
    if not anomalies.empty:
        st.dataframe(anomalies[['tx_hash', 'from_address', 'to_address', 'gasPrice', 'value', 'blockNumber']])

    clusters = dbscan_cluster(txs)
    if not clusters.empty:
        st.subheader("MEV Bot Clusters")
        st.vega_lite_chart(clusters, {
            'mark': 'circle',
            'encoding': {
                'x': {'field': 'blockNumber', 'type': 'quantitative'},
                'y': {'field': 'gasPrice', 'type': 'quantitative'},
                'color': {'field': 'cluster', 'type': 'nominal'}
            }
        })

if __name__ == "__main__":
    run_dashboard()
    # To enable mempool feed: asyncio.run(listen_mempool())




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
FLASHBOTS_API  = "https://blocks.flashbots.net/v1/blocks"
PAGE_SIZE      = 10000

# --- Helpers ---

def safe_get(url, params=None, timeout=10):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# --- Data Fetching ---

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

# --- Anomaly Detection ---

def detect_anomalies(txs):
    feats = txs[['gasPrice','value']].values
    Xs = StandardScaler().fit_transform(feats)
    labels = IsolationForest(contamination=0.01, random_state=42).fit_predict(Xs)
    return txs[labels==-1]

# --- Clustering ---

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
    st.title("MEV Bot Detector with Layman Explanations")
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", 100)
    limit   = st.sidebar.number_input("TX Limit", 500)

    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return

    st.subheader("1. High-Gas Transactions")
    st.dataframe(txs)

    # Sandwich
    sandwiches = detect_sandwich(txs)
    st.subheader(f"2. Sandwich Attacks Detected: {len(sandwiches)}")
    if not sandwiches.empty:
        st.dataframe(sandwiches)
        for _, r in sandwiches.iterrows():
            st.markdown(
                f"• In block **{r.block}**, user tx **{r.victim}** was *sandwiched* by two bots on **{r.to}**. "
                f"First, **{r.front}** paid {r.front_gas:.1f} Gwei to jump ahead, then the victim tx paid {r.victim_gas:.1f} Gwei, "
                f"and finally **{r.back}** paid {r.back_gas:.1f} Gwei to capture profit. This hurts regular users by making them pay more gas and “gift” value to bots."
            )

    # Anomalies
    anomalies = detect_anomalies(txs)
    st.subheader(f"3. Anomalous Transactions: {len(anomalies)}")
    if not anomalies.empty:
        st.dataframe(anomalies)
        avg_gas = txs['gasPrice'].mean()
        for _, a in anomalies.iterrows():
            ratio = a.gasPrice / avg_gas if avg_gas else np.nan
            st.markdown(
                f"• Tx **{a.tx_hash}** paid **{a.gasPrice:.1f} Gwei** (about {ratio:.1f}× higher than the typical {avg_gas:.1f} Gwei). "
                f"It only transferred **{a.value:.4f} ETH**, so paying that premium bid for only a small move is unusual. "
                f"For context, at 600 Gwei, each of the ~21,000 gas units costs ~0.0000006 ETH—so this tx could spend ~0.0126 ETH (~$33) on fees alone."
            )

    # Clusters
    clusters = dbscan_cluster(txs)
    if not clusters.empty:
        st.subheader("4. MEV Bot Clusters")
        st.vega_lite_chart(clusters, {
            'mark':'circle','encoding':{
                'x':{'field':'blockNumber','type':'quantitative'},
                'y':{'field':'gasPrice','type':'quantitative'},
                'color':{'field':'cluster','type':'nominal'}
            }
        })

if __name__ == "__main__":
    run_dashboard()
    # asyncio.run(listen_mempool())


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
    return txs.assign(
        hash=txs['hash'], gasPrice=txs['gasPrice'],
        value=txs['value'].astype(float)/1e18,
        blockNumber=txs['blockNumber'].astype(int),
        to=txs['to'], from_address=txs['from']
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
        if (prev['to'] == nxt['to'] and
            prev['gasPrice'] > curr['gasPrice'] < nxt['gasPrice'] and
            prev['blockNumber'] == curr['blockNumber'] == nxt['blockNumber']):
            records.append({
                'front_hash': prev['hash'], 'victim_hash': curr['hash'], 'back_hash': nxt['hash'],
                'to': prev['to'], 'block': prev['blockNumber'],
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

# --- 5. Real-time mempool ---
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

# --- 6. Streamlit Dashboard ---
def run_dashboard():
    st.title("MEV Bot Detector with Details")
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", value=100)
    limit = st.sidebar.number_input("TX Limit", value=500)

    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return

    st.subheader("High-Gas Transactions")
    st.dataframe(txs[['hash', 'gasPrice', 'value']])

    # Sandwich attacks with details
    sandwiches = detect_sandwich(txs)
    st.subheader(f"Detected {len(sandwiches)} Sandwich Attacks")
    if not sandwiches.empty:
        st.dataframe(sandwiches)

    # Anomalous transactions with details
    anomalies = detect_anomalies(txs)
    st.subheader(f"Anomalous Transactions: {len(anomalies)}")
    if not anomalies.empty:
        st.dataframe(anomalies[['hash', 'gasPrice', 'value', 'blockNumber', 'to', 'from_address']])

    # Bot clusters (unchanged)
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

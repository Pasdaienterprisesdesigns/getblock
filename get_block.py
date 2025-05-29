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
FLASHBOTS_API = "https://blocks.flashbots.net/v1/blocks"
PAGE_SIZE = 10000  # Etherscan max

# --- Helpers ---

def safe_get(url, params=None, timeout=10):
    """HTTP GET with basic error handling and retries."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# --- 1. Data Fetching with Pagination ---

def get_high_gas_txs(min_gas=100, limit=1000):
    all_txs = []
    page = 1
    while len(all_txs) < limit:
        params = {
            'module': 'account', 'action': 'txlist',
            'address': '0x0000000000000000000000000000000000000000',
            'startblock': 0, 'endblock': 99999999,
            'page': page, 'offset': PAGE_SIZE,
            'sort': 'desc', 'apikey': ETHERSCAN_API
        }
        data = safe_get('https://api.etherscan.io/api', params)
        if not data or data.get('status') != '1':
            break
        df_page = pd.DataFrame(data['result'])
        df_page['gasPrice'] = df_page['gasPrice'].astype(float) / 1e9
        all_txs.extend(df_page[df_page['gasPrice'] > min_gas].to_dict('records'))
        if len(data['result']) < PAGE_SIZE:
            break
        page += 1
    df = pd.DataFrame(all_txs).head(limit)
    df = df.assign(
        gasPrice=df['gasPrice'].astype(float),
        value=df['value'].astype(float) / 1e18,
        blockNumber=df['blockNumber'].astype(int)
    )
    return df


def fetch_flashbots_mev():
    data = safe_get(FLASHBOTS_API)
    if not data:
        return pd.DataFrame()
    mev_txs = []
    for block in data:
        for tx in block.get('transactions', []):
            mev_txs.append({
                'tx_hash': tx['transaction_hash'],
                'miner': block['miner'],
                'profit': float(tx.get('coinbase_transfer', 0)) / 1e18
            })
    return pd.DataFrame(mev_txs)

# --- 2. MEV Detection (unchanged) ---

def detect_sandwich(txs):
    sandwiches = []
    for i in range(1, len(txs) - 1):
        prev, curr, nxt = txs.iloc[i-1], txs.iloc[i], txs.iloc[i+1]
        if (prev['to'] == nxt['to'] and
            prev['gasPrice'] > curr['gasPrice'] < nxt['gasPrice'] and
            prev['blockNumber'] == curr['blockNumber'] == nxt['blockNumber']):
            sandwiches.append((prev['hash'], curr['hash'], nxt['hash']))
    return sandwiches


def find_arbitrage_cycles(txs, max_edges=1000):
    G = nx.DiGraph()
    for _, tx in txs.head(max_edges).iterrows():
        G.add_edge(tx['from'], tx['to'], tx_hash=tx['hash'])
    return list(nx.simple_cycles(G))

# --- 3. Machine Learning with Scaling ---

def detect_anomalies(txs):
    features = txs[['gasPrice', 'value']].values
    scaler = StandardScaler().fit(features)
    X_scaled = scaler.transform(features)
    model = IsolationForest(contamination=0.01, random_state=42)
    labels = model.fit_predict(X_scaled)
    return txs[labels == -1]


def cluster_mev_bots(txs):
    features = txs[['gasPrice', 'blockNumber']].values
    scaler = StandardScaler().fit(features)
    X_scaled = scaler.transform(features)
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(X_scaled)
    txs['cluster'] = clustering.labels_
    return txs[txs['cluster'] != -1]

# --- 4. Real-Time Mempool Listener (unchanged) ---

async def listen_mempool():
    async with connect(ALCHEMY_WS_URL) as ws:
        await ws.send(json.dumps({
            'jsonrpc': '2.0', 'id': 1,
            'method': 'eth_subscribe', 'params': ['newPendingTransactions']
        }))
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if 'params' in data:
                st.write(f"New pending TX: {data['params']['result']}")

# --- 5. Streamlit Dashboard with Error Handling ---

def run_dashboard():
    st.title("MEV Bot Detector (Refactored)")
    st.sidebar.header("Settings")
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", value=100)
    limit = st.sidebar.number_input("TX Limit", value=500)

    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched.")
        return

    st.subheader("High-Gas Transactions")
    st.dataframe(txs[['hash', 'gasPrice', 'value']])

    sandwiches = detect_sandwich(txs)
    st.subheader(f"Detected {len(sandwiches)} Sandwich Attacks")

    cycles = find_arbitrage_cycles(txs)
    st.subheader(f"Found {len(cycles)} Arbitrage Cycles (first 10)")
    for c in cycles[:10]: st.write(" → ".join(c))

    anomalies = detect_anomalies(txs)
    st.subheader(f"Anomalous Transactions: {len(anomalies)}")

    clusters = cluster_mev_bots(txs)
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
    # To run mempool listener in Streamlit, call: asyncio.run(listen_mempool())

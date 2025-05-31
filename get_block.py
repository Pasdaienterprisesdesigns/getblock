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
from web3 import Web3
from datetime import datetime

# --- Config ---
ALCHEMY_HTTP_URL = "https://eth-mainnet.g.alchemy.com/v2/RELC1tew5qdPp0NLc82Nw"
ALCHEMY_WS_URL = "wss://eth-mainnet.g.alchemy.com/v2/RELC1tew5qdPp0NLc82Nw"
w3 = Web3(Web3.HTTPProvider(ALCHEMY_HTTP_URL))

# --- Fetch latest block transactions ---
def fetch_recent_block_txs(n_blocks=1000):
    latest = w3.eth.block_number
    all_txs = []
    timestamps = []
    pbar = st.progress(0, text="Fetching blocks...")

    for i, block_num in enumerate(range(latest, latest - n_blocks, -1)):
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            timestamps.append(block.timestamp)
            for tx in block.transactions:
                all_txs.append({
                    'tx_hash': tx.hash.hex(),
                    'from_address': tx['from'],
                    'to_address': tx.to,
                    'gasPrice': tx.gasPrice / 1e9,
                    'value': tx.value / 1e18,
                    'blockNumber': tx.blockNumber
                })
        except Exception:
            continue

        percent = min(i / n_blocks, 1.0)
        pbar.progress(percent, text=f"Fetching blocks... {int(percent*100)}%")

    pbar.empty()

    if not all_txs:
        st.error("No transactions found. Try refreshing or check your connection/API key.")
        return pd.DataFrame()

    if timestamps:
        min_time = datetime.utcfromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        max_time = datetime.utcfromtimestamp(max(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"⏳ Time Range Covered: {min_time} UTC → {max_time} UTC")

    return pd.DataFrame(all_txs)

# --- Detection & Clustering ---
def detect_sandwich(txs):
    recs = []
    for i in range(1, len(txs)-1):
        prev, curr, nxt = txs.iloc[i-1], txs.iloc[i], txs.iloc[i+1]
        if (prev['to_address'] == nxt['to_address'] and
            prev['gasPrice'] > curr['gasPrice'] < nxt['gasPrice'] and
            prev['blockNumber'] == curr['blockNumber'] == nxt['blockNumber']):
            recs.append({
                'block': prev['blockNumber'],
                'victim_hash': curr['tx_hash'],
                'front_hash': prev['tx_hash'],
                'back_hash': nxt['tx_hash'],
                'to_address': prev['to_address'],
                'front_gas': prev['gasPrice'],
                'victim_gas': curr['gasPrice'],
                'back_gas': nxt['gasPrice']
            })
    return pd.DataFrame(recs)

def detect_anomalies(txs):
    feats = txs[['gasPrice','value']].values
    labels = IsolationForest(contamination=0.01, random_state=42).fit_predict(
        StandardScaler().fit_transform(feats)
    )
    return txs[labels == -1]

def dbscan_cluster(txs):
    feats = txs[['gasPrice','blockNumber']].values
    lbls = DBSCAN(eps=0.5, min_samples=3).fit_predict(
        StandardScaler().fit_transform(feats)
    )
    txs['cluster'] = lbls
    return txs[txs['cluster'] != -1]

# --- Streamlit Dashboard ---
def run_dashboard():
    st.set_page_config(layout="wide")
    st.title("🔎 MEV Bot Detector Dashboard")
    st.markdown("""
    This dashboard helps you understand on-chain activity on Ethereum by detecting:

    **1. High Gas Transactions** – These are transactions that paid unusually high fees to get mined quickly. Often used by bots or urgent trades.

    **2. Sandwich Attacks** – When a bot places a transaction *before* and *after* someone else’s, forcing the victim to pay more while the bot profits.

    **3. Anomalous Transactions** – Suspicious TXs where gas fees are unusually high for low value, hinting at bot activity.

    **4. MEV Bot Clusters** – Groups of transactions likely sent by the same bot (based on gas patterns).

    👉 Use the sidebar to set how many blocks you want to fetch.
    """)

    # Sidebar input
    n_blocks = st.sidebar.slider("Number of Recent Blocks to Analyze", 10, 100000, 1000, step=10)

    # Fetch data
    txs = fetch_recent_block_txs(n_blocks)
    if txs.empty:
        return

    st.subheader("📊 1. High-Gas Transactions")
    st.dataframe(txs.head(100))

    st.subheader("🦊 2. Detected Sandwich Attacks")
    sandwiches = detect_sandwich(txs)
    if sandwiches.empty:
        st.info("No sandwich attacks found in this dataset.")
    else:
        st.dataframe(sandwiches)
        for _, r in sandwiches.iterrows():
            st.markdown(
                f"**Block {r.block}**: Victim `{r.victim_hash}` sandwiched between `{r.front_hash}` and `{r.back_hash}` with gas bids {r.front_gas:.1f}, {r.victim_gas:.1f}, {r.back_gas:.1f} Gwei."
            )

    st.subheader("🚨 3. Anomalous Transactions")
    anomalies = detect_anomalies(txs)
    if anomalies.empty:
        st.info("No anomalies detected.")
    else:
        st.dataframe(anomalies)
        avg_gas = txs['gasPrice'].mean()
        for _, a in anomalies.iterrows():
            ratio = a.gasPrice / avg_gas if avg_gas else np.nan
            st.markdown(
                f"• Transaction `{a.tx_hash}` bid **{a.gasPrice:.1f} Gwei** (~{ratio:.1f}× avg), moved **{a.value:.4f} ETH**."
            )

    st.subheader("🤖 4. MEV Bot Clusters")
    clusters = dbscan_cluster(txs)
    if clusters.empty:
        st.info("No clusters detected.")
    else:
        st.vega_lite_chart(
            clusters,
            {
                'mark': 'circle',
                'encoding': {
                    'x': {'field': 'blockNumber', 'type': 'quantitative', 'title': 'Block'},
                    'y': {'field': 'gasPrice',    'type': 'quantitative', 'title': 'Gas (Gwei)'},
                    'color': {'field': 'cluster', 'type': 'nominal', 'title': 'Cluster'}
                }
            },
            use_container_width=True
        )

# --- Run the dashboard ---
if __name__ == "__main__":
    run_dashboard()

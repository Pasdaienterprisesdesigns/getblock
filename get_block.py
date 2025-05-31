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
ETHERSCAN_API = "972W1N6UZ2IC6MXZJ32G7JJJT4UNMRNP6B"
ALCHEMY_WS_URL = "https://eth-mainnet.g.alchemy.com/v2/RELC1tew5qdPp0NLc82Nw"
ALCHEMY_WS_URL = "wss://eth-mainnet.g.alchemy.com/v2/v2/RELC1tew5qdPp0NLc82Nw"
PAGE_SIZE      = 10000
w3 = Web3(Web3.HTTPProvider(ALCHEMY_HTTP_URL))

# --- Helpers ---
def fetch_recent_block_txs(n_blocks=10000):
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

    if timestamps:
        min_time = datetime.utcfromtimestamp(min(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        max_time = datetime.utcfromtimestamp(max(timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"Time Range Covered: {min_time} UTC to {max_time} UTC")

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

# --- Real-time mempool (optional) ---
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

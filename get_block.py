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
PAGE_SIZE      = 10000

# --- Helpers ---

def safe_get(url, params=None, timeout=10):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None

# --- Fetch Etherscan TXs ---
def get_high_gas_txs(min_gas=100, limit=1000):
    all_txs = []
    page = 1
    pbar = st.progress(0, text="Fetching transactions...")

    while len(all_txs) < limit:
        percent = min(len(all_txs) / limit, 1.0)
        pbar.progress(percent, text=f"Fetching transactions... {int(percent*100)}%")

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

    pbar.empty()

    txs = pd.DataFrame(all_txs).head(limit)
    txs['gasPrice'] = txs['gasPrice'].astype(float)
    txs['value'] = txs['value'].astype(float) / 1e18
    txs['blockNumber'] = txs['blockNumber'].astype(int)
    txs = txs.rename(columns={'hash':'tx_hash', 'from':'from_address', 'to':'to_address'})
    return txs[['tx_hash','from_address','to_address','gasPrice','value','blockNumber']]

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

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
                'victim': curr['tx_hash'],
                'front': prev['tx_hash'],
                'back': nxt['tx_hash'],
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

# --- Streamlit Dashboard ---
def run_dashboard():
    # Introduction for lay users
    st.title("MEV Bot Detector")
    st.markdown(
        """
        **What is this dashboard?**
        This tool helps you spot unusual Ethereum transactions and potential MEV (Miner Extractable Value) bots.
        - **High-Gas Transactions**: shows transactions that paid more gas than normal.
        - **Sandwich Attacks**: detects when bots sandwich a user’s transaction, forcing them to pay extra.
        - **Anomalous Transactions**: flags TXs that bid unusually high gas for small transfers.
        - **MEV Bot Clusters**: groups similar transactions to reveal bot activity patterns.

        **How to use**: adjust the **Min Gas** and **TX Limit** on the sidebar, then explore each section to understand who paid extra gas, why it’s odd, and which bots might be active.
        """
    )
    min_gas = st.sidebar.number_input("Min Gas (Gwei)", value=100)
    limit = st.sidebar.number_input("TX Limit", value=500)

    # 1. High-Gas Transactions
    txs = get_high_gas_txs(min_gas, limit)
    if txs.empty:
        st.warning("No transactions fetched. Try increasing the limit or lowering Min Gas.")
        return
    st.subheader("1. High-Gas Transactions")
    st.dataframe(txs)

    # 2. Sandwich Attacks
    sandwiches = detect_sandwich(txs)
    st.subheader(f"2. Sandwich Attacks Detected: {len(sandwiches)}")
    if sandwiches.empty:
        st.write("No sandwich attacks found in the sample.")
    else:
        for _, r in sandwiches.iterrows():
            st.markdown(
                f"**In block {r.block}:** a user transaction (**{r.victim}**) got squeezed between two bots. "
                f"A bot front-ran it (**{r.front}**) at {r.front_gas:.1f} Gwei, then back-ran (**{r.back}**) at {r.back_gas:.1f} Gwei."
            )

    # 3. Anomalous Transactions
    anomalies = detect_anomalies(txs)
    st.subheader(f"3. Anomalous Transactions: {len(anomalies)}")
    if anomalies.empty:
        st.write("No outlier transactions detected.")
    else:
        avg_gas = txs['gasPrice'].mean()
        for _, a in anomalies.iterrows():
            ratio = a.gasPrice / avg_gas if avg_gas else np.nan
            st.markdown(
                f"• **{a.tx_hash}** bid {a.gasPrice:.1f} Gwei (~{ratio:.1f}× average), moving only {a.value:.4f} ETH. "
                "This mismatch flags it as unusual."
            )

    # 4. MEV Bot Clusters
    clusters = dbscan_cluster(txs)
    st.subheader("4. MEV Bot Clusters")
    if clusters.empty:
        st.write("No bot clusters found.")
    else:
        st.vega_lite_chart(
            clusters,
            {
                'mark': 'circle',
                'encoding': {
                    'x': {'field': 'blockNumber', 'type': 'quantitative', 'title': 'Block Number'},
                    'y': {'field': 'gasPrice',    'type': 'quantitative', 'title': 'Gas Price (Gwei)'},
                    'color': {'field': 'cluster', 'type': 'nominal',       'title': 'Cluster ID'}
                }
            },
            use_container_width=True
        )

if __name__ == "__main__":
    run_dashboard()
    # To listen to real-time mempool: asyncio.run(listen_mempool())

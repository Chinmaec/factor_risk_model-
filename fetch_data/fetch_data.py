
import time
import requests
import pandas as pd
import os

API_KEY = os.getenv("EODHD_API_KEY", "") 
BASE_URL = "https://eodhd.com/api/eod"  # EODHD daily price endpoint


# Edit this list to choose symbols to download
TICKERS = [
    "AAPL.US", "MSFT.US", "NVDA.US",
    "JPM.US", "GS.US", "BAC.US",
    "JNJ.US", "UNH.US",
    "AMZN.US", "MCD.US", 
    "XOM.US", "CVX.US", 
    "CAT.US", "HON.US", 
    "T.US",
]

# Fixed backtest window; make dynamic if needed
start_date = "2025-05-11"
end_date = "2026-05-11"

frames = []
for i, ns_ticker in enumerate(TICKERS):
    eodhd_ticker = ns_ticker.replace(".NS", ".NSE")
    url = (
        f"{BASE_URL}/{eodhd_ticker}"
        f"?api_token={API_KEY}"
        f"&from={start_date}&to={end_date}"
        f"&period=d&adjusted=1&fmt=json"
    )

    print(f"[{i+1:02d}] req symbol={eodhd_ticker} url={url}")

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"[{i+1:02d}] {ns_ticker} HTTP {resp.status_code} body={resp.text[:300]}")
            continue
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            print(f"[{i+1:02d}/20] SKIP {ns_ticker} — API error: {data['error']}")
            continue
        if not data:
            print(f"[{i+1:02d}/20] SKIP {ns_ticker} — empty response")
            continue
        series = pd.Series(
            [row["adjusted_close"] for row in data],
            index=pd.to_datetime([row["date"] for row in data]),
            name=ns_ticker,
        )
        frames.append(series)
        print(f"[{i+1:02d}/20] OK   {ns_ticker} — {len(series)} rows")
    except Exception as e:
        print(f"[{i+1:02d}/20] ERROR {ns_ticker} — {e}")
    time.sleep(0.5)

if frames:
    prices = pd.concat(frames, axis=1).sort_index()
    os.makedirs("data", exist_ok=True)
    prices.to_csv("data/prices.csv")
    print(f"\nSaved: {prices.shape[0]} rows × {prices.shape[1]} tickers → data/prices.csv")
    print(prices.tail(3))
else:
    print("No data downloaded. Check your API key.")
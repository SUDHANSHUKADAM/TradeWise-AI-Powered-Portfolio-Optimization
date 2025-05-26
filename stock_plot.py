from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import mplfinance as mpf
import time

ALPHA_VANTAGE_KEY = "W3WF05YZJQPCA97A"  # 🔁 Replace with your real API key
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

def fetch_alpha_vantage_candle(ticker):
    try:
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        return data.tail(14)  # Last 2 weeks
    except Exception as e:
        print(f"❌ Error fetching {ticker} from Alpha Vantage: {e}")
        return None

def plot_stock_candlestick(tickers):
    for ticker in tickers:
        print(f"📊 Plotting Alpha Vantage candlestick for {ticker}...")
        df = fetch_alpha_vantage_candle(ticker)
        if df is not None and not df.empty:
            mpf.plot(df, type="candle", style="charles",
                     title=f"{ticker} - Last 2 Weeks (Alpha Vantage)",
                     ylabel="Price ($)", volume=False)
        else:
            print(f"⚠️ Skipping {ticker}, no data found.")
        time.sleep(15)  # Alpha Vantage allows 5 requests/minute







import requests
import time

API_KEY = "d0ns7npr01qn5ghln47gd0ns7npr01qn5ghln480"  # replace with your actual key

def get_current_prices(tickers):
    base_url = "https://finnhub.io/api/v1/quote"
    prices = {}
    for ticker in tickers:
        try:
            response = requests.get(base_url, params={"symbol": ticker, "token": API_KEY})
            data = response.json()
            price = data.get("c")  # 'c' = current price
            prices[ticker] = round(price, 2) if price else None
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            prices[ticker] = None
        time.sleep(0.2)  # small delay to stay well below the rate limit
    return prices


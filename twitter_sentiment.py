import os
import json
import time
import requests
from datetime import datetime, timezone
from finbert_sentiment import get_sentiment


BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJfQ1wEAAAAANsGX%2FRj80KHgkXY%2BDF81YHOyglk%3DdQrEMB7G8vm1ascdgN5hQm0B6SCAvmsfzZnZNvBRz0TWDZGcqh"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "JPM", "JNJ",
    "UNH", "V", "XOM", "PG", "MA", "HD", "LLY", "ABBV", "PEP", "CVX",
    "AVGO", "KO", "MRK", "ADBE", "COST", "WMT", "BAC", "DIS", "TMO", "CRM",
    "MCD", "INTC", "NFLX", "CSCO", "VZ", "PFE", "ABT", "ACN", "LIN", "NKE",
    "AMD", "DHR", "NEE", "TXN", "PM", "WFC", "BMY", "UNP", "QCOM", "UPS",
    "MS", "ORCL", "AMGN", "RTX", "INTU", "SCHW", "LOW", "SPGI", "NOW", "HON",
    "AMT", "IBM", "CAT", "GS", "DE", "LMT", "GE", "AXP", "CB", "BA",
    "MDT", "ISRG", "BLK", "ADI", "ZTS", "T", "PLD", "SYK", "C", "EL",
    "MO", "CI", "GILD", "MDLZ", "ADP", "BDX", "MMC", "PNC", "REGN", "ADSK",
    "SO", "GM", "DUK", "EW", "CL", "CSX", "SHW", "ICE", "FIS", "FTNT"
]


CACHE_FILE = "twitter_sentiments.json"
MAX_TWEETS_PER_REQUEST = 10
REQUEST_DELAY = 75  # In seconds
HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"}
TWITTER_URL = "https://api.twitter.com/2/tweets/search/recent"
LIMITED_MODE = False  # Set True if you only want to update 2-3 tickers for quick testing

def load_twitter_sentiments(tickers=None):
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            if tickers:
                return {ticker: cache.get(ticker, {}).get("sentiment", 0) for ticker in tickers}
            return cache
    except:
        return {ticker: 0 for ticker in tickers} if tickers else {}

def fetch_tweets(ticker):
    query = f"{ticker} stock lang:en -is:retweet"
    params = {
        "query": query,
        "max_results": MAX_TWEETS_PER_REQUEST,
        "tweet.fields": "text"
    }
    try:
        response = requests.get(TWITTER_URL, headers=HEADERS, params=params)
        if response.status_code == 429:
            print(f"⏳ Rate limited. Waiting 15 minutes before retrying for {ticker}...")
            time.sleep(900)
            return fetch_tweets(ticker)
        elif response.status_code != 200:
            print(f"❌ Error for {ticker}: {response.status_code} - {response.text}")
            return []
        tweets = [tweet["text"] for tweet in response.json().get("data", [])]
        # Filter empty or short tweets
        return [t for t in tweets if len(t.strip()) > 15]
    except Exception as e:
        print(f"❌ Exception for {ticker}: {e}")
        return []

def analyze_sentiment(tweets):
    if not tweets:
        return 0
    return get_sentiment(tweets)

def build_sentiment_cache():
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

    timestamp = datetime.now(timezone.utc).isoformat()

    tickers_to_process = TICKERS[:3] if LIMITED_MODE else TICKERS

    for ticker in tickers_to_process:
        if ticker in cache:
            print(f"⚠️ {ticker} already cached. Skipping.")
            continue

        print(f"\n🔍 Fetching tweets and analyzing sentiment for {ticker}...")
        tweets = fetch_tweets(ticker)
        score = analyze_sentiment(tweets)
        cache[ticker] = {"sentiment": score, "timestamp": timestamp}

        print(f"✅ {ticker} sentiment: {score} | Tweets used: {len(tweets)}")

        # Save after each ticker
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

        time.sleep(REQUEST_DELAY)

    print(f"\n✅ All data saved to {CACHE_FILE}")

if __name__ == "__main__":
    build_sentiment_cache()



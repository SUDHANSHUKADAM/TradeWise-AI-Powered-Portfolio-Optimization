import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
)

from forecast_engine import forecast_prices
from finbert_sentiment import compare_sentiment_models
from reddit_sentiment import init_reddit, get_sentiment_score as reddit_score
from news_sentiment import get_news_sentiment
from twitter_sentiment import load_twitter_sentiments
from recommender import generate_recommendations
from stock_price import get_current_prices

API_KEY = "5a60bb6928684365aba30061c8226bff"

TOP_100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "LLY", "ABBV", "PEP",
    "KO", "MRK", "BAC", "AVGO", "TMO", "ADBE", "COST", "WMT", "CRM", "ABT",
    "INTC", "ACN", "DHR", "QCOM", "TXN", "MCD", "LIN", "NEE", "WFC", "AMGN",
    "VZ", "NKE", "MS", "ORCL", "PM", "HON", "UNP", "MDT", "UPS", "IBM",
    "RTX", "SCHW", "GS", "BLK", "ISRG", "CVS", "AMT", "SBUX", "DE", "PLD",
    "T", "INTU", "ELV", "SPGI", "CI", "CAT", "GE", "SYK", "MO", "C",
    "ADP", "MMC", "MDLZ", "ZTS", "TJX", "LRCX", "CB", "SO", "PNC", "DUK",
    "USB", "ADI", "CL", "GILD", "BDX", "PGR", "REGN", "BKNG", "VRTX", "TGT",
    "APD", "FDX", "CSCO", "LOW", "ECL", "PXD", "WM", "MET", "TRV", "AEP"
]

def get_actual_prices(ticker):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "outputsize": 40,
        "apikey": API_KEY
    }
    try:
        response = requests.get(url, params=params).json()
        if "values" not in response:
            return []
        df = pd.DataFrame(response["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df["close"].astype(float).tolist()
    except:
        return []

def evaluate_all():
    reddit = init_reddit()
    twitter_scores = load_twitter_sentiments()
    tickers = TOP_100_TICKERS[:8]  # limit for speed

    total_lstm = total_sentiment = total_rec = 0
    count_lstm = count_sentiment = count_rec = 0

    total_reddit = total_twitter = total_news = 0
    count_reddit = count_twitter = count_news = 0

    original_correct = 0
    finetuned_correct = 0
    total_model_count = 0

    print("\n🚀 Starting evaluation for 8 tickers...\n")

    for ticker in tickers:
        try:
            actual_prices = get_actual_prices(ticker)
            if len(actual_prices) < 3:
                continue

            direction_actual = actual_prices[-1] > actual_prices[-2]

            # -------------------- LSTM --------------------
            forecast = forecast_prices(ticker)
            if forecast:
                direction_pred = forecast["30d"] > actual_prices[-2]
                total_lstm += int(direction_pred == direction_actual)
                count_lstm += 1

            # -------------------- Reddit --------------------
            reddit_s = reddit_score(reddit, ticker)
            pred_reddit = reddit_s > 0
            total_reddit += int(pred_reddit == direction_actual)
            count_reddit += 1

            # -------------------- Twitter --------------------
            twitter_s = twitter_scores.get(ticker, 0)
            print(f"[DEBUG] {ticker} → Twitter Score: {twitter_s:.3f}")
            pred_twitter = twitter_s > 0
            total_twitter += int(pred_twitter == direction_actual)
            count_twitter += 1

            # -------------------- News --------------------
            news_s = get_news_sentiment(ticker)
            pred_news = news_s > 0
            total_news += int(pred_news == direction_actual)
            count_news += 1

            # -------------------- Combined Sentiment --------------------
            combined_score = 0.5 * news_s + 0.4 * reddit_s + 0.1 * twitter_s
            predicted_up = combined_score > 0.05
            total_sentiment += int(predicted_up == direction_actual)
            count_sentiment += 1

            # -------------------- Recommendation --------------------
            prices = get_current_prices([ticker])
            if prices and ticker in prices:
                yesterday_price = actual_prices[-2]
                recommendations = generate_recommendations(
                    {ticker: 1}, prices, {ticker: combined_score}, 1000,
                    {ticker: reddit_s}, {ticker: twitter_s}, {ticker: news_s},
                    {ticker: yesterday_price}
                )
                rec_action = recommendations[ticker]["action"]
                actual_action = "BUY" if prices[ticker] > yesterday_price else "SELL"
                if rec_action in ["BUY", "SELL"]:
                    total_rec += int(rec_action == actual_action)
                    count_rec += 1

            # -------------------- FinBERT Comparison --------------------
            example_text = f"{ticker} stock performed this week"
            comparison = compare_sentiment_models(example_text)[0]

            for model_key in ["original", "finetuned"]:
                predicted_score = comparison[model_key]["score"]
                direction_pred = predicted_score > 0
                correct = direction_pred == direction_actual
                if model_key == "original":
                    original_correct += int(correct)
                else:
                    finetuned_correct += int(correct)
                total_model_count += 1

        except Exception as e:
            print(f"❌ Error with {ticker}: {e}")

    print("\n📊 Summary Across 8 Tickers:")

    if count_lstm > 0:
        print(f"✅ LSTM Direction Accuracy: {100 * total_lstm / count_lstm:.2f}% ({count_lstm} tickers)")
    if count_sentiment > 0:
        print(f"✅ Combined Sentiment Accuracy: {100 * total_sentiment / count_sentiment:.2f}% ({count_sentiment} tickers)")
    if count_reddit > 0:
        print(f"📘 Reddit Sentiment Accuracy: {100 * total_reddit / count_reddit:.2f}% ({count_reddit} tickers)")
    if count_twitter > 0:
        print(f"🐦 Twitter Sentiment Accuracy: {100 * total_twitter / count_twitter:.2f}% ({count_twitter} tickers)")
    if count_news > 0:
        print(f"📰 News Sentiment Accuracy: {100 * total_news / count_news:.2f}% ({count_news} tickers)")
    if count_rec > 0:
        print(f"🎯 Recommendation Accuracy: {100 * total_rec / count_rec:.2f}% ({count_rec} tickers)")

    if total_model_count > 0:
        print(f"\n🔹 Original FinBERT Accuracy: {100 * original_correct / (total_model_count // 2):.2f}%")
        print(f"🔸 Fine-Tuned FinBERT Accuracy: {100 * finetuned_correct / (total_model_count // 2):.2f}%")


def evaluate_finbert_accuracy():
    sources = {
        "Twitter": [
            ("Great earnings report boosts stock", "positive"),
            ("Stock crashes due to bad management", "negative"),
            ("Investors await Fed decision", "neutral")
        ],
        "Reddit": [
            ("Company shows massive growth", "positive"),
            ("Layoffs announced this morning", "negative"),
            ("Stock remains unchanged for weeks", "neutral")
        ],
        "News": [
            ("CEO arrested for fraud", "negative"),
            ("Revenue beats expectations", "positive"),
            ("Markets closed for holiday", "neutral")
        ]
    }

    print("\n📊 FinBERT Classification Accuracy:")
    for source, data in sources.items():
        texts, labels = zip(*data)
        original_preds = [compare_sentiment_models(text)[0]["original"]["label"] for text in texts]
        finetuned_preds = [compare_sentiment_models(text)[0]["finetuned"]["label"] for text in texts]

        print(f"\n🔹 {source} (Original):")
        print(classification_report(labels, original_preds, digits=3))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, original_preds))

        print(f"\n🔸 {source} (Fine-Tuned):")
        print(classification_report(labels, finetuned_preds, digits=3))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, finetuned_preds))

if __name__ == "__main__":
    print("Device set to use cpu")
    evaluate_all()
    evaluate_finbert_accuracy()

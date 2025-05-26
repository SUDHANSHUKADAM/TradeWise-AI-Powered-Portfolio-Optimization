import warnings
warnings.filterwarnings("ignore")

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

from stock_price import get_current_prices
from reddit_sentiment import init_reddit, get_sentiment_score
from news_sentiment import get_news_sentiment
from recommender import generate_recommendations
from stock_plot import plot_stock_candlestick
from twitter_sentiment import CACHE_FILE as TWITTER_CACHE_FILE
from alert_engine import detect_event_alerts
from forecast_engine import forecast_prices, forecast_with_lstm
from suggest_stocks import suggest_stocks_to_buy
from finbert_sentiment import compare_sentiment_models
import requests
import json
import copy

TWELVE_API_KEY = "5a60bb6928684365aba30061c8226bff"


def get_user_portfolio():
    portfolio = {}
    try:
        n = int(input("📊 How many different stocks do you own? "))
        for i in range(n):
            ticker = input(f"Enter stock symbol #{i+1} (e.g., AAPL): ").upper()
            qty = int(input(f"Enter quantity of {ticker}: "))
            portfolio[ticker] = qty
        budget = float(input("💰 Enter your available budget in dollars (e.g., 2000): "))
        return portfolio, budget
    except ValueError:
        print("❌ Invalid input. Please enter numbers for quantity and budget.")
        return {}, 0


def load_twitter_sentiments(tickers):
    twitter_sentiments = {}
    try:
        with open(TWITTER_CACHE_FILE, "r") as f:
            cache = json.load(f)
        for ticker in tickers:
            twitter_sentiments[ticker] = cache.get(ticker, {}).get("sentiment", 0)
    except FileNotFoundError:
        twitter_sentiments = {ticker: 0 for ticker in tickers}
    return twitter_sentiments


def combine_sentiments_with_finbert(tickers, reddit, twitter, news, model_choice='original', weights=(0.2, 0.2, 0.2, 0.4)):
    combined = {}
    for ticker in tickers:
        r = reddit.get(ticker, 0)
        t = twitter.get(ticker, 0)
        n = news.get(ticker, 0)
        text = f"{ticker} stock activity this week"
        comparison = compare_sentiment_models(text)[0]
        f_score = comparison[model_choice]['score']
        combined[ticker] = round(r * weights[0] + t * weights[1] + n * weights[2] + f_score * weights[3], 3)
        print(f"\n📌 {ticker} - Using {model_choice} FinBERT Sentiment")
        print(f"Reddit: {r}, Twitter: {t}, News: {n}, FinBERT: {f_score} → Combined: {combined[ticker]}")
    return combined


def get_yesterday_close(ticker):
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": 2,
            "apikey": TWELVE_API_KEY
        }
        response = requests.get(url, params=params).json()
        if "values" in response and len(response["values"]) >= 2:
            return float(response["values"][1]["close"])
    except:
        return None


def run_full_pipeline(label, portfolio, budget, model_choice):
    print(f"\n====== RUNNING FULL PIPELINE USING {label.upper()} FINBERT ======")

    prices = get_current_prices(portfolio.keys())

    print("\n📊 Current Holdings Value:")
    total_value = 0
    for ticker, qty in portfolio.items():
        price = prices.get(ticker)
        if price:
            value = price * qty
            print(f"{ticker} ({qty} shares): ${value:.2f}")
            total_value += value

    print(f"\n💼 Total Portfolio Value (excluding budget): ${total_value:.2f}")

    reddit = init_reddit()
    reddit_sentiments = {ticker: get_sentiment_score(reddit, ticker) for ticker in portfolio}
    twitter_sentiments = load_twitter_sentiments(portfolio)
    news_sentiments = {ticker: get_news_sentiment(ticker) for ticker in portfolio}

    final_sentiments = combine_sentiments_with_finbert(portfolio, reddit_sentiments, twitter_sentiments, news_sentiments, model_choice=model_choice)
    yesterday_closes = {ticker: get_yesterday_close(ticker) for ticker in portfolio}

    recommendations = generate_recommendations(
        portfolio, prices, final_sentiments, budget,
        reddit_sentiments, twitter_sentiments, news_sentiments, yesterday_closes
    )

    print("\n📈 Recommendations:")
    for ticker, rec in recommendations.items():
        emoji = "🟢" if rec['action'] == "BUY" else "🔴" if rec['action'] == "SELL" else "🟡"
        print(f"{emoji} {ticker}: {rec['action']}")
        print(f"    Reason: {rec['reason']}")
        print(f"    Explanation: {rec['explanation']}")

    print("\n📉 Candlestick Chart:")
    plot_stock_candlestick(portfolio.keys())

    print("\n🔔 Alerts:")
    alert_triggered = False
    for ticker in portfolio:
        current = prices.get(ticker)
        yesterday = yesterday_closes.get(ticker)
        if current and yesterday:
            change = ((current - yesterday) / yesterday) * 100
            symbol = "📈" if change > 0 else "📉" if change < 0 else "➖"
            print(f"{symbol} {ticker} moved {change:+.2f}% since yesterday's close.")
            alert_triggered = True
    if not alert_triggered:
        print("✅ No alert-worthy events found.")

    print("\n📈 LSTM-Based Forecast:")
    for ticker in portfolio:
        try:
            forecast_with_lstm(ticker)
        except Exception:
            pass

    total_budget = budget + sum(prices[t] * rec['current_quantity'] for t, rec in recommendations.items() if rec['action'] == "SELL")
    suggestions, best_combo, combo_profit = suggest_stocks_to_buy(total_budget, portfolio)

    print("\n🛒 Buy Suggestions Based on Available Budget:")
    if best_combo:
        for s in best_combo:
            growth_7d = round(s['forecast_7d'] - s['price'], 2)
            growth_30d = round(s['forecast_30d'] - s['price'], 2)
            print(f"📌 {s['ticker']}")
            print(f"    Price: ${s['price']}")
            print(f"    Suggested Quantity: {s['quantity']}")
            print(f"    Total Cost: ${s['total_cost']:.2f}")
            print(f"    Forecast 7 Days: ${s['forecast_7d']} 📈 (+${growth_7d})")
            print(f"    Forecast 30 Days: ${s['forecast_30d']} 📈 (+${growth_30d})")
            print(f"    Reason: Strong LSTM forecasted rise in short and long term.")
    else:
        print("✅ No strong buy suggestions found within current budget.")

    print("\n🧾 Final Portfolio Recommendation:")
    for ticker, rec in recommendations.items():
        if rec["action"] == "HOLD":
            print(f"🟡 Hold: {ticker}")

    if best_combo:
        print(f"💰 Based on your remaining ${total_budget:.0f} budget, you *can choose to buy*:")
        for s in best_combo:
            potential_profit = round((s['forecast_30d'] - s['price']) * s['quantity'], 2)
            print(f"   - {s['ticker']} ({s['quantity']} shares) for ~${s['total_cost']:.0f} → 📈 30-day profit: ~${potential_profit}")


if __name__ == '__main__':
    portfolio, budget = get_user_portfolio()

    print("\n✅ Portfolio Summary:")
    print("📦 Stocks:", portfolio)
    print("💰 Budget: $", budget)

    run_full_pipeline("Original", copy.deepcopy(portfolio), budget, model_choice='original')
    run_full_pipeline("FineTuned", copy.deepcopy(portfolio), budget, model_choice='finetuned')










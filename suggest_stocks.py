import itertools
import json
from stock_price import get_current_prices
from news_sentiment import get_news_sentiment
from reddit_sentiment import init_reddit, get_sentiment_score
from twitter_sentiment import CACHE_FILE as TWITTER_CACHE_FILE
from forecast_engine import forecast_prices

def load_twitter_sentiments():
    try:
        with open(TWITTER_CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def get_combined_sentiment(ticker, reddit, twitter_cache):
    reddit_score = get_sentiment_score(reddit, ticker)
    twitter_score = twitter_cache.get(ticker, {}).get("sentiment", 0)
    news_score = get_news_sentiment(ticker)
    combined = round((0.3 * reddit_score + 0.4 * twitter_score + 0.3 * news_score), 3)
    return {
        "reddit": reddit_score,
        "twitter": twitter_score,
        "news": news_score,
        "combined": combined
    }

def suggest_stocks_to_buy(user_budget, user_portfolio):
    top_100 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ", "V", "XOM", "PG",
        "MA", "HD", "UNH", "PFE", "BAC", "KO", "MRK", "DIS", "WMT", "INTC", "PEP", "CVX", "NFLX",
        "ADBE", "T", "ABT", "CSCO", "CRM", "COST", "NKE", "MCD", "ORCL", "QCOM", "LLY", "IBM",
        "TXN", "GE", "BMY", "UPS", "DHR", "AVGO", "MDT", "NEE", "PM", "AXP", "LIN", "TMO",
        "UNP", "GS", "HON", "MS", "C", "CAT", "AMGN", "ISRG", "ADP", "DE", "LMT", "ZTS", "PLD",
        "NOW", "BLK", "SPGI", "CI", "MMC", "SYK", "GILD", "MO", "EL", "PNC", "BDX", "SO", "DUK",
        "REGN", "ETN", "ROST", "CB", "SHW", "VRTX", "FISV", "CL", "ADI", "APD", "CSX", "HUM",
        "FTNT", "MNST", "CTAS", "AON", "KMB", "STZ", "MAR", "ADM", "ECL", "VLO", "WELL"
    ]

    reddit = init_reddit()
    twitter_cache = load_twitter_sentiments()
    candidates = []

    for ticker in top_100:
        if ticker in user_portfolio:
            continue
        try:
            sentiment = get_combined_sentiment(ticker, reddit, twitter_cache)
            forecast = forecast_prices(ticker)
            if forecast is None:
                continue

            current_price = forecast["current"]
            forecast_7d = forecast["7d"]
            forecast_30d = forecast["30d"]
            growth_7d = forecast_7d - current_price
            growth_30d = forecast_30d - current_price

            if growth_30d > 0 or sentiment["combined"] > 0.2:
                candidates.append({
                    "ticker": ticker,
                    "price": current_price,
                    "forecast_7d": forecast_7d,
                    "forecast_30d": forecast_30d,
                    "growth_7d_dollars": growth_7d,
                    "growth_30d_dollars": growth_30d,
                    "sentiment": sentiment
                })
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue

    best_combo = []
    max_profit = -float("inf")

    for r in range(1, 4):  # try combos of 1 to 3
        for combo in itertools.combinations(candidates, r):
            total_cost, total_profit = 0, 0
            combo_detail = []

            for stock in combo:
                max_qty = int((user_budget - total_cost) // stock["price"])
                if max_qty == 0:
                    continue

                cost = max_qty * stock["price"]
                profit = max_qty * stock["growth_30d_dollars"]
                profit_7d = max_qty * stock["growth_7d_dollars"]

                total_cost += cost
                total_profit += profit

                combo_detail.append({
                    "ticker": stock["ticker"],
                    "price": round(stock["price"], 2),
                    "quantity": max_qty,
                    "total_cost": round(cost, 2),
                    "forecast_7d": round(stock["forecast_7d"], 2),
                    "forecast_30d": round(stock["forecast_30d"], 2),
                    "profit_7d": round(profit_7d, 2),
                    "profit_30d": round(profit, 2),
                    "reason": "Strong LSTM forecasted rise in short and long term."
                })

            if total_cost <= user_budget and total_profit > max_profit:
                max_profit = total_profit
                best_combo = combo_detail

    # Top 5 individual stock suggestions
    top_suggestions = sorted(candidates, key=lambda x: x["growth_30d_dollars"], reverse=True)[:5]
    suggestions_dict = {}

    for stock in top_suggestions:
        qty = int(user_budget // stock["price"])
        total_cost = qty * stock["price"]
        suggestions_dict[stock["ticker"]] = {
            "price": round(stock["price"], 2),
            "suggested_quantity": qty,
            "total_cost": round(total_cost, 2),
            "forecast_7d": round(stock["forecast_7d"], 2),
            "forecast_30d": round(stock["forecast_30d"], 2),
            "growth_7d_dollars": round(qty * stock["growth_7d_dollars"], 2),
            "growth_30d_dollars": round(qty * stock["growth_30d_dollars"], 2),
            "sentiment": stock["sentiment"]["combined"],
            "reddit": stock["sentiment"]["reddit"],
            "twitter": stock["sentiment"]["twitter"],
            "news": stock["sentiment"]["news"],
            "reason": f"Strong LSTM forecasted rise in short and long term."
        }

    # Fallback: If no best combo found, choose top 1 from suggestions_dict
    if not best_combo and suggestions_dict:
        top = next(iter(suggestions_dict.items()))
        fallback = {
            "ticker": top[0],
            "price": top[1]["price"],
            "quantity": top[1]["suggested_quantity"],
            "total_cost": top[1]["total_cost"],
            "forecast_7d": top[1]["forecast_7d"],
            "forecast_30d": top[1]["forecast_30d"],
            "profit_7d": top[1]["growth_7d_dollars"],
            "profit_30d": top[1]["growth_30d_dollars"],
            "reason": top[1]["reason"]
        }
        best_combo = [fallback]
        max_profit = fallback["profit_30d"]

    return suggestions_dict, best_combo, round(max_profit, 2)








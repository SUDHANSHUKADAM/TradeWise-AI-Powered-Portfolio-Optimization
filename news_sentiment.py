import requests
from datetime import datetime, timedelta
from finbert_sentiment import get_sentiment

API_KEY = "d0ns7npr01qn5ghln47gd0ns7npr01qn5ghln47gd0ns7npr01qn5ghln480"
BASE_URL = "https://finnhub.io/api/v1/company-news"

def get_news_sentiment(ticker, max_articles=5, return_articles=False, debug=False):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')

    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            if debug:
                print(f"📭 No recent news found for {ticker}.")
            return (0, []) if return_articles else 0

        texts = []
        selected_articles = []

        for article in data[:max_articles]:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline} {summary}".strip()
            if text:
                texts.append(text)
                selected_articles.append({
                    "headline": headline,
                    "summary": summary
                })

        sentiment_score = get_sentiment(texts)

        if sentiment_score == 0 and debug:
            print(f"😐 All news for {ticker} evaluated as neutral by FinBERT.")

        return (sentiment_score, selected_articles) if return_articles else sentiment_score

    except Exception as e:
        if debug:
            print(f"❌ Error fetching news for {ticker}: {e}")
        return (0, []) if return_articles else 0





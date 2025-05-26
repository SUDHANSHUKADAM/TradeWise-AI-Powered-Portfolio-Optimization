from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load FinBERT model
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

# Trigger keywords for finance-related events
TRIGGER_KEYWORDS = [
    "earnings", "plunge", "drop", "cuts", "missed", "beats", "loss", "profit",
    "scandal", "layoffs", "fired", "merger", "acquisition", "bankruptcy", "recall",
    "jump", "surge", "record", "spike", "explode", "revenue", "guidance"
]

def finbert_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze()
        labels = ["negative", "neutral", "positive"]
        return dict(zip(labels, probs.tolist()))

def detect_event_alerts(ticker, news_articles, current_price, yesterday_price):
    alerts = []

    if not current_price or not yesterday_price:
        return alerts

    percent_change = ((current_price - yesterday_price) / yesterday_price) * 100

    for article in news_articles:
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        full_text = (headline + " " + summary).lower()

        if any(keyword in full_text for keyword in TRIGGER_KEYWORDS):
            sentiment_probs = finbert_score(full_text)
            dominant = max(sentiment_probs, key=sentiment_probs.get)
            prob = sentiment_probs[dominant]

            if abs(percent_change) >= 3 and prob >= 0.7 and dominant != "neutral":
                alert = (
                    f"⚠️ {ticker} likely affected by {dominant.upper()} news: "
                    f"\"{headline.strip()}\""
                )
                alerts.append(alert)

    return alerts




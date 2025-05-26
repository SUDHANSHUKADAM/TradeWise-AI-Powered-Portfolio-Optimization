def generate_recommendations(portfolio, prices, sentiments, budget, reddit_sentiments, twitter_sentiments, news_sentiments, yesterday_closes):
    recommendations = {}
    threshold_buy = 0.3
    threshold_sell = -0.3

    for ticker, qty in portfolio.items():
        sentiment = sentiments.get(ticker, 0)
        reddit = reddit_sentiments.get(ticker, 0)
        twitter = twitter_sentiments.get(ticker, 0)
        news = news_sentiments.get(ticker, 0)
        price = prices.get(ticker, None)
        yesterday = yesterday_closes.get(ticker, None)

        if price is None:
            recommendations[ticker] = {
                "action": "HOLD",
                "reason": "Price unavailable",
                "suggested_quantity": 0,
                "current_quantity": qty,
                "explanation": f"No real-time price data found for {ticker}."
            }
            continue

        percent_change = None
        if yesterday:
            percent_change = ((price - yesterday) / yesterday) * 100

        # Generate explanation
        explanation = []
        if percent_change is not None:
            explanation.append(f"Price moved {percent_change:+.2f}% since yesterday.")

        explanation.append(f"Sentiment Scores - Reddit: {reddit}, Twitter: {twitter}, News: {news}.")

        aligned_positive = all(s > 0.2 for s in [reddit, twitter, news])
        aligned_negative = all(s < -0.2 for s in [reddit, twitter, news])

        if sentiment >= threshold_buy and aligned_positive and percent_change is not None and percent_change < 0:
            reason = "High market optimism despite recent dip"
            explanation.append("All sources show optimism while price fell — indicating possible rebound.")
            action = "BUY"
            suggested_quantity = int(budget // price)

        elif sentiment <= threshold_sell and aligned_negative and percent_change is not None and percent_change > 0:
            reason = "Negative sentiment despite price rise"
            explanation.append("Price increased but sentiment is bearish — indicating overvaluation risk.")
            action = "SELL"
            suggested_quantity = 0

        elif abs(sentiment) < 0.2:
            reason = "Unclear market signals"
            explanation.append("Sentiment scores are mixed or neutral. No confident movement expected.")
            action = "HOLD"
            suggested_quantity = 0

        else:
            reason = "Moderate signals"
            if sentiment > 0:
                explanation.append("Sentiment is slightly positive, holding for confirmation.")
            else:
                explanation.append("Slight bearish tone, but not enough evidence for selling.")
            action = "HOLD"
            suggested_quantity = 0

        recommendations[ticker] = {
            "action": action,
            "reason": reason,
            "suggested_quantity": suggested_quantity,
            "current_quantity": qty,
            "explanation": " ".join(explanation)
        }

    return recommendations


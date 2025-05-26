import praw
from finbert_sentiment import get_sentiment

# Reddit API credentials
REDDIT_CLIENT_ID = "3F-UTLp3jjB2d7cRYYwx8g"
REDDIT_SECRET = "moeQjaji46B-ipwV5T0IrC1xysLQoA"
REDDIT_USER_AGENT = "portfolio_sentiment_app"

def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def get_sentiment_score(reddit, ticker, post_limit=10, debug=False):
    try:
        posts = reddit.subreddit("stocks+investing+wallstreetbets").search(ticker, limit=post_limit)
        texts = [post.title + " " + post.selftext for post in posts if post.title or post.selftext]
        if not texts:
            return 0
        return get_sentiment(texts)
    except Exception as e:
        if debug:
            print(f"❌ Error analyzing sentiment for {ticker}: {e}")
        return 0



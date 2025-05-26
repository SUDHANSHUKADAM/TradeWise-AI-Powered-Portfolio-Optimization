# 📈 TradeWise: AI-Powered Portfolio Optimization

## 📄 Abstract

The Intelligent Real-Time Portfolio Optimizer, branded as TradeWise, is an AI-powered platform designed to empower users with informed financial decisions by integrating live stock market data, social media sentiment, and deep learning models. It functions as a personalized assistant, offering buy, sell, or hold recommendations based on portfolio performance and market dynamics. This system leverages advanced natural language processing through sentiment analysis models like FinBERT and integrates time series forecasting using LSTM networks to predict future stock trends. The platform is deployed on Google Cloud for seamless scalability and real-time operation, laying a robust foundation for the future of AI-driven investing.

## 📝 Introduction

The dynamic and volatile nature of modern financial markets makes it increasingly important for investors to adopt intelligent tools for decision-making. Traditional portfolio management solutions often overlook real-time data and public sentiment, which can significantly influence stock prices. TradeWise addresses this gap by incorporating real-time sentiment analysis from social platforms and predictive modeling to offer timely, personalized investment recommendations.

### Key Objectives:
- To provide a reliable and scalable platform for real-time stock portfolio analysis.
- To personalize recommendations based on user holdings, budget, and market trends.
- To analyze sentiment from sources like Twitter, Reddit, and financial news.
- To employ deep learning models for accurate price forecasting.
- To visualize data through interactive charts and intuitive outputs.

Phase 1 of the project focuses on optimizing portfolios involving 100 companies using a combination of fine-tuned sentiment analysis and LSTM-based forecasting models.

## 🔧 Technologies and Tools

### Software and Frameworks:
- **Python**: Core development language.
- **Google Cloud Platform**: For hosting and deployment.
- **VS Code**: Preferred development environment.

### Machine Learning Libraries:
- `transformers`: Used for sentiment analysis with FinBERT.
- `PyTorch`, `TensorFlow`, `Keras`: For deep learning and time series forecasting.
- `scikit-learn`: For classical ML models and evaluation.
- `XGBoost`: For ensemble learning and comparison.

### APIs and Data Sources:
- `yfinance`: For historical and live stock price data.
- `alpha_vantage`: To collect OHLC and volume data.
- `Twitter API v2`: To gather real-time tweet data.
- `Reddit PRAW`: For Reddit posts and discussion scraping.
- `Finnhub`: For real-time financial news and summaries.

### Data Handling and Visualization:
- `pandas`, `numpy`: For data manipulation.
- `matplotlib`, `plotly`: For stock candlestick and trend plotting.

## 🚀 Proposed System

The TradeWise system functions as a comprehensive pipeline consisting of several interconnected modules. These include:

- **Portfolio Intake Module**: Allows users to enter their current holdings and investment budget.
- **Sentiment Analysis Module**: Utilizes fine-tuned FinBERT to analyze Reddit, Twitter, and news headlines.
- **Time Series Forecasting Module**: Implements LSTM models to forecast price trends for the next 3–5 days.
- **Recommendation Engine**: Combines sentiment and forecasting outputs with user portfolio to suggest buy/sell/hold actions.
- **Visualization Dashboard**: Presents candlestick plots, forecasted trends, sentiment histograms, and tabulated recommendations.

## ⚙️ System Architecture

```mermaid
graph TD
    A[User Portfolio Input] --> B[Real-Time Data Fetcher]
    B --> C[Reddit/Twitter/News Collector]
    B --> D[Price History & Live Quotes]
    C --> E[Fine-Tuned FinBERT Sentiment Analysis]
    D --> F[Candlestick Chart + LSTM Forecasting]
    E --> H[Buy/Sell/Hold Decision Engine]
    F --> H
    H --> I[Final Recommendation Output]

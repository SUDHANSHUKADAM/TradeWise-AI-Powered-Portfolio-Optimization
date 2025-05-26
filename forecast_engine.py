import requests
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lstm_model import StockLSTM
from datetime import datetime

# Hyperparameters
SEQ_LENGTH = 30
FUTURE_DAYS = 30
API_KEY = "5a60bb6928684365aba30061c8226bff"

def get_twelve_data(ticker, days=120):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": "1day",
        "outputsize": days,
        "apikey": API_KEY
    }
    response = requests.get(url, params=params).json()

    if "values" not in response:
        print(f"❌ Failed to get data for {ticker}: {response}")
        return None

    df = pd.DataFrame(response["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df["close"] = df["close"].astype(float)
    return df

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length:i + seq_length + 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model(X, y):
    model = StockLSTM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for _ in range(50):
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train[:, -1, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, device

def forecast_prices(ticker):
    df = get_twelve_data(ticker)
    if df is None or len(df) < SEQ_LENGTH + FUTURE_DAYS:
        return None

    close_prices = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = create_sequences(scaled, SEQ_LENGTH)
    model, device = train_lstm_model(X, y)

    model.eval()
    input_seq = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)

    predictions = []
    for _ in range(FUTURE_DAYS):
        with torch.no_grad():
            next_val = model(input_seq)
        predictions.append(next_val.cpu().numpy().flatten()[0])
        input_seq = torch.cat([input_seq[:, 1:, :], next_val.unsqueeze(0)], dim=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    current_price = float(close_prices[-1])
    forecast_7d = round(forecast[6], 2)
    forecast_30d = round(forecast[-1], 2)
    return {
        "current": current_price,
        "7d": forecast_7d,
        "30d": forecast_30d,
        "7d_gain": round(forecast_7d - current_price, 2),
        "30d_gain": round(forecast_30d - current_price, 2)
    }

def forecast_with_lstm(ticker):
    df = get_twelve_data(ticker)
    if df is None or len(df) < SEQ_LENGTH + FUTURE_DAYS:
        print(f"❌ Not enough data for {ticker}")
        return

    close_prices = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = create_sequences(scaled, SEQ_LENGTH)
    model, device = train_lstm_model(X, y)

    model.eval()
    input_seq = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)

    predictions = []
    for _ in range(FUTURE_DAYS):
        with torch.no_grad():
            next_val = model(input_seq)
        predictions.append(next_val.cpu().numpy().flatten()[0])
        input_seq = torch.cat([input_seq[:, 1:, :], next_val.unsqueeze(0)], dim=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS)

    # Plot forecast
    plt.figure(figsize=(12, 5))
    plt.plot(df.index[-60:], close_prices[-60:], label="Past Prices")
    plt.plot(future_dates, forecast, label="Forecast", marker='o')
    plt.title(f"{ticker} - LSTM Forecast (Next {FUTURE_DAYS} days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Forecast Summary for {ticker}:")
    print(f"➡️  Next 7 Days: ${forecast[6]:.2f} 📈 (+${forecast[6] - close_prices[-1][0]:.2f})")
    print(f"➡️  Next 30 Days: ${forecast[-1]:.2f} 📈 (+${forecast[-1] - close_prices[-1][0]:.2f})")





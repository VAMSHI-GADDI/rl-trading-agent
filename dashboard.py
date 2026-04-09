
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from stable_baselines3 import PPO
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

st.set_page_config(page_title="RL Trading Agent", layout="wide")
st.title("RL Trading Agent - Live Dashboard")
st.caption("PPO-based agent making real-time decisions on live market data")

def get_latest_data(ticker, days=60):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    df.reset_index(inplace=True)
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["macd"] = ta.trend.macd(df["close"])
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["bb_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["bb_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    return df.dropna().reset_index(drop=True)

def get_prediction(ticker):
    df = get_latest_data(ticker)
    model = PPO.load("ppo_trading_agent")
    latest = df.iloc[-1]
    obs = np.array([[
        float(latest["close"]), float(latest["volume"]),
        float(latest["macd"]), float(latest["rsi"]),
        float(latest["bb_upper"]), float(latest["bb_lower"]),
        float(latest["atr"]), 1.0, 0.0
    ]], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return {
        "date": str(latest["date"])[:10],
        "price": round(float(latest["close"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "macd": round(float(latest["macd"]), 4),
        "bb_upper": round(float(latest["bb_upper"]), 2),
        "bb_lower": round(float(latest["bb_lower"]), 2),
        "action": action_map[int(np.array(action).flatten()[0])],
        "history": df[["date", "close", "rsi", "macd"]].tail(30)
    }

ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])

if st.sidebar.button("Get Live Prediction"):
    with st.spinner("Fetching live market data and running agent..."):
        result = get_prediction(ticker)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stock", ticker)
    col2.metric("Price", f"${result['price']}")
    col3.metric("RSI", result["rsi"])
    col4.metric("MACD", result["macd"])

    action = result["action"]
    color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[action]
    st.markdown(f"<h1 style='text-align:center; color:{color}'>Agent Decision: {action}</h1>", unsafe_allow_html=True)

    st.subheader("Recent Price History")
    st.line_chart(result["history"].set_index("date")["close"])

    st.subheader("RSI (Last 30 Days)")
    st.line_chart(result["history"].set_index("date")["rsi"])

    st.subheader("Raw Market Data")
    st.dataframe(result["history"])

    st.caption(f"Last updated: {result['date']}")
else:
    st.info("Select a stock from the sidebar and click Get Live Prediction")

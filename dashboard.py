
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
st.caption("PPO-based agent — auto-retrains on latest market data every session")

def get_latest_data(ticker, days=500):
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

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.feature_cols = ["close", "volume", "macd", "rsi", "bb_upper", "bb_lower", "atr"]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_cols) + 2,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.returns = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        cash_ratio = self.balance / self.initial_balance
        share_ratio = (self.shares_held * self.df.iloc[self.current_step]["close"]) / self.initial_balance
        return np.append(row, [cash_ratio, share_ratio])

    def step(self, action):
        price = self.df.iloc[self.current_step]["close"]
        prev_worth = self.net_worth
        if action == 1 and self.balance >= price:
            shares = self.balance // price
            self.shares_held += shares
            self.balance -= shares * price
        elif action == 2 and self.shares_held > 0:
            self.balance += self.shares_held * price
            self.shares_held = 0
        self.net_worth = self.balance + self.shares_held * price
        daily_return = (self.net_worth - prev_worth) / (prev_worth + 1e-9)
        self.returns.append(daily_return)
        reward = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) >= 2 else 0.0
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

def retrain_model(ticker):
    df = get_latest_data(ticker)
    train_df = df.iloc[:int(len(df) * 0.8)].reset_index(drop=True)
    env = StockTradingEnv(train_df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    model.save("ppo_trading_agent")
    return df

def get_prediction(df, model):
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

with st.spinner("Fetching latest market data and retraining model..."):
    df = retrain_model(ticker)
    model = PPO.load("ppo_trading_agent")
    result = get_prediction(df, model)

st.success("Model retrained on latest data. Showing today's prediction.")

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

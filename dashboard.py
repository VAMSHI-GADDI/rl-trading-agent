
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from stable_baselines3 import PPO
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient

API_KEY    = "PKOIZMUEO7LP2ANCCXHFB35Q2S"
SECRET_KEY = "GrDQfCZg5rtsbPWCVovwZNqAnDLuUGmcyvVGqhcovbno"

st.set_page_config(page_title="RL Trading Agent", layout="wide")
st.title("RL Trading Agent - Live Dashboard")
st.caption("PPO-based agent with live Alpaca paper trading")

@st.cache_data(ttl=300)
def get_alpaca_account():
    try:
        client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        orders = client.get_orders()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "positions": [{"symbol": p.symbol, "qty": float(p.qty), "market_value": float(p.market_value), "unrealized_pl": float(p.unrealized_pl)} for p in positions],
            "orders": [{"symbol": o.symbol, "side": str(o.side), "qty": float(o.qty), "status": str(o.status), "created_at": str(o.created_at)[:19]} for o in list(orders)[:10]]
        }
    except Exception as e:
        return {"error": str(e)}

def get_prediction(ticker):
    try:
        end = datetime.today()
        start = end - timedelta(days=60)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        df.reset_index(inplace=True)
        df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        df = df.sort_values("date").reset_index(drop=True)
        df["macd"] = ta.trend.macd(df["close"])
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["bb_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
        df["bb_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df = df.dropna().reset_index(drop=True)
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
            "action": action_map[int(np.array(action).flatten()[0])],
            "history": df[["date", "close", "rsi"]].tail(30)
        }
    except Exception as e:
        return {"error": str(e)}

# Sidebar
st.sidebar.title("Controls")
ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
run_prediction = st.sidebar.button("Get Live Prediction")

# Portfolio section
st.subheader("Live Alpaca Paper Portfolio")
acct = get_alpaca_account()

if "error" in acct:
    st.error(f"Alpaca connection error: {acct['error']}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}",
                delta=f"${acct['portfolio_value']-100000:,.2f} vs start")
    col2.metric("Cash", f"${acct['cash']:,.2f}")
    col3.metric("Buying Power", f"${acct['buying_power']:,.2f}")

    if acct["positions"]:
        st.subheader("Current Positions")
        pos_df = pd.DataFrame(acct["positions"])
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("No open positions yet — run paper_trader.py to place orders")

    if acct["orders"]:
        st.subheader("Recent Orders")
        ord_df = pd.DataFrame(acct["orders"])
        st.dataframe(ord_df, use_container_width=True)

st.divider()

# Prediction section
if run_prediction:
    with st.spinner(f"Fetching live data for {ticker}..."):
        result = get_prediction(ticker)

    if "error" in result:
        st.error(result["error"])
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stock", ticker)
        col2.metric("Price", f"${result['price']}")
        col3.metric("RSI", result["rsi"])
        col4.metric("MACD", result["macd"])

        action = result["action"]
        color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[action]
        st.markdown(f"<h1 style='text-align:center; color:{color}'>Agent Decision: {action}</h1>",
                    unsafe_allow_html=True)

        st.subheader("Price History (Last 30 Days)")
        st.line_chart(result["history"].set_index("date")["close"])

        st.subheader("RSI (Last 30 Days)")
        st.line_chart(result["history"].set_index("date")["rsi"])

        st.caption(f"Last updated: {result['date']}")
else:
    st.info("Select a stock and click Get Live Prediction")

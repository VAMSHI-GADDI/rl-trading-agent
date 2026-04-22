from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import yfinance as yf
import numpy as np
import ta
from datetime import datetime, timedelta

API_KEY    = "PKOIZMUEO7LP2ANCCXHFB35Q2S"
SECRET_KEY = "GrDQfCZg5rtsbPWCVovwZNqAnDLuUGmcyvVGqhcovbno"

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_signal(ticker):
    df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df['rsi']  = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])
    latest = df.iloc[-1]
    rsi  = float(latest['rsi'])
    macd = float(latest['macd'])
    # Simple rule: RSI < 40 and MACD negative = BUY, RSI > 60 = SELL
    if rsi < 40 and macd < 0:
        return "BUY", float(latest['close'])
    elif rsi > 60:
        return "SELL", float(latest['close'])
    return "HOLD", float(latest['close'])

def place_order(ticker, side, qty=1):
    order = client.submit_order(
        MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
    )
    return order

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

account = client.get_account()
print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
print(f"Buying power:    ${float(account.buying_power):,.2f}\n")

for ticker in TICKERS:
    signal, price = get_signal(ticker)
    print(f"{ticker}: ${price:.2f} → {signal}")
    if signal == "BUY":
        order = place_order(ticker, OrderSide.BUY, qty=1)
        print(f"  ✓ BUY order placed: {order.id}")
    elif signal == "SELL":
        positions = {p.symbol: p for p in client.get_all_positions()}
        if ticker in positions:
            order = place_order(ticker, OrderSide.SELL, qty=1)
            print(f"  ✓ SELL order placed: {order.id}")
        else:
            print(f"  — No position to sell")
    else:
        print(f"  — Holding")
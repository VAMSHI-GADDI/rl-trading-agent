from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import ta
from stable_baselines3 import PPO
from auth import (get_db, authenticate_user, register_user, create_token,
                  User, Portfolio, ACCESS_TOKEN_EXPIRE_MINUTES,
                  oauth2_scheme, SECRET_KEY, ALGORITHM)
from jose import JWTError, jwt
import sendgrid
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = "YOUR_SENDGRID_KEY"  # get free at sendgrid.com
FROM_EMAIL       = "your@email.com"

app = FastAPI(title="RL Trading Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic schemas ---
class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

class TradeLog(BaseModel):
    ticker: str
    action: str
    price: float
    qty: int
    pnl: float = 0.0

# --- Auth helpers ---
def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Email alert ---
def send_alert(to_email: str, ticker: str, action: str, price: float):
    try:
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=to_email,
            subject=f"RL Trading Alert: {action} {ticker}",
            html_content=f"""
            <h2>Trading Signal Alert</h2>
            <p>Your RL agent has generated a signal:</p>
            <h1 style='color:{"green" if action=="BUY" else "red"}'>{action} {ticker}</h1>
            <p>Current Price: <strong>${price}</strong></p>
            <p>Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
            <hr>
            <small>RL Trading Agent — automated signal</small>
            """
        )
        sg.send(message)
    except Exception as e:
        print(f"Email error: {e}")

# --- Prediction helper ---
def get_prediction(ticker: str):
    df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df["macd"]     = ta.trend.macd(df["close"])
    df["rsi"]      = ta.momentum.rsi(df["close"], window=14)
    df["bb_upper"] = ta.volatility.bollinger_hband(df["close"], window=20)
    df["bb_lower"] = ta.volatility.bollinger_lband(df["close"], window=20)
    df["atr"]      = ta.volatility.average_true_range(
                         df["high"], df["low"], df["close"], window=14)
    df = df.dropna()
    latest = df.iloc[-1]
    model  = PPO.load("ppo_trading_agent")
    obs    = np.array([[
        float(latest["close"]), float(latest["volume"]),
        float(latest["macd"]),  float(latest["rsi"]),
        float(latest["bb_upper"]), float(latest["bb_lower"]),
        float(latest["atr"]),   1.0, 0.0
    ]], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return {
        "ticker": ticker,
        "price":  round(float(latest["close"]), 2),
        "rsi":    round(float(latest["rsi"]), 2),
        "macd":   round(float(latest["macd"]), 4),
        "action": action_map[int(np.array(action).flatten()[0])]
    }

# --- Routes ---
@app.get("/")
def root():
    return {"status": "RL Trading Agent API is live"}

@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    user = register_user(db, req.email, req.username, req.password)
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return {"message": f"Welcome {user.username}!", "user_id": user.id}

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(
        {"sub": user.email},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": token, "token_type": "bearer"}

@app.get("/predict/{ticker}")
def predict(ticker: str, background_tasks: BackgroundTasks,
            current_user: User = Depends(get_current_user),
            db: Session = Depends(get_db)):
    result = get_prediction(ticker)
    # Send email alert for BUY/SELL
    if result["action"] != "HOLD":
        background_tasks.add_task(
            send_alert, current_user.email,
            ticker, result["action"], result["price"]
        )
    return result

@app.post("/portfolio/log")
def log_trade(trade: TradeLog,
              current_user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    entry = Portfolio(
        user_id=current_user.id,
        ticker=trade.ticker,
        action=trade.action,
        price=trade.price,
        qty=trade.qty,
        pnl=trade.pnl
    )
    db.add(entry)
    db.commit()
    return {"message": "Trade logged"}

@app.get("/portfolio")
def get_portfolio(current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    trades = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id
    ).order_by(Portfolio.timestamp.desc()).all()
    return [
        {"ticker": t.ticker, "action": t.action,
         "price": t.price, "qty": t.qty,
         "pnl": t.pnl, "time": str(t.timestamp)[:19]}
        for t in trades
    ]

@app.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db)):
    users  = db.query(User).all()
    board  = []
    for user in users:
        trades = db.query(Portfolio).filter(
            Portfolio.user_id == user.id).all()
        total_pnl = sum(t.pnl for t in trades)
        total_trades = len(trades)
        if total_trades > 0:
            board.append({
                "username":     user.username,
                "total_pnl":    round(total_pnl, 2),
                "total_trades": total_trades,
                "joined":       str(user.created_at)[:10]
            })
    return sorted(board, key=lambda x: x["total_pnl"], reverse=True)
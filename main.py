
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO

app = FastAPI(title="RL Trading Agent API")
model = PPO.load("ppo_trading_agent")

class MarketSnapshot(BaseModel):
    close: float
    volume: float
    macd: float
    rsi: float
    bb_upper: float
    bb_lower: float
    atr: float
    cash_ratio: float
    share_ratio: float

@app.get("/")
def root():
    return {"status": "RL Trading Agent is live"}

@app.post("/predict")
def predict(data: MarketSnapshot):
    obs = np.array([[
        data.close, data.volume, data.macd, data.rsi,
        data.bb_upper, data.bb_lower, data.atr,
        data.cash_ratio, data.share_ratio
    ]], dtype=np.float32)
    
    action, _ = model.predict(obs, deterministic=True)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    return {
        "action": action_map[int(action)],
        "action_code": int(action)
    }

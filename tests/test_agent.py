import numpy as np
from stable_baselines3 import PPO

def test_model_loads():
    model = PPO.load("ppo_trading_agent")
    assert model is not None

def test_model_predicts():
    model = PPO.load("ppo_trading_agent")
    obs = np.random.rand(1, 9).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in [0, 1, 2]

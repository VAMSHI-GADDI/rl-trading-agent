# RL Trading Agent

A PPO-based reinforcement learning agent that learns to trade stocks, with a full MLOps pipeline.

## Results
| Metric | PPO Agent | Buy and Hold |
|--------|-----------|----------|
| Final Value | $15,342 | $15,318 |
| Sharpe Ratio | 0.5201 | 0.5180 |

## Stack
- RL: Stable-Baselines3 (PPO)
- Environment: Custom OpenAI Gym
- Tracking: MLflow
- API: FastAPI
- Container: Docker
- CI/CD: GitHub Actions

## Quickstart
```
pip install -r requirements.txt
uvicorn main:app --reload
```

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY    = "PKOIZMUEO7LP2ANCCXHFB35Q2S"
SECRET_KEY = "GrDQfCZg5rtsbPWCVovwZNqAnDLuUGmcyvVGqhcovbno"

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Place a test buy order
order = client.submit_order(
    MarketOrderRequest(
        symbol="AAPL",
        qty=2,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
)

print(f"✓ Order placed!")
print(f"  Symbol: {order.symbol}")
print(f"  Side:   {order.side}")
print(f"  Qty:    {order.qty}")
print(f"  Status: {order.status}")
print(f"  ID:     {order.id}")
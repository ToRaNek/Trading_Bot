"""Test Alpaca Paper Trading Connection"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("TEST ALPACA PAPER TRADING")
print("=" * 50)

# Check env vars
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
base_url = os.getenv("ALPACA_BASE_URL")

print(f"\nAPI Key: {api_key[:10]}..." if api_key else "API Key: NOT SET")
print(f"Secret Key: {secret_key[:10]}..." if secret_key else "Secret Key: NOT SET")
print(f"Base URL: {base_url}")

try:
    import alpaca_trade_api as tradeapi

    # Connect
    print("\n[1] Connecting to Alpaca...")
    api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

    # Get account
    print("[2] Getting account info...")
    account = api.get_account()
    print(f"    Status: {account.status}")
    print(f"    Cash: ${float(account.cash):,.2f}")
    print(f"    Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"    Buying Power: ${float(account.buying_power):,.2f}")

    # Check if market is open
    print("\n[3] Checking market status...")
    clock = api.get_clock()
    print(f"    Market Open: {clock.is_open}")
    print(f"    Next Open: {clock.next_open}")
    print(f"    Next Close: {clock.next_close}")

    # Get positions
    print("\n[4] Current positions...")
    positions = api.list_positions()
    if positions:
        for p in positions:
            print(f"    {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f}")
    else:
        print("    No positions")

    # Try to buy 1 share of AAPL
    print("\n[5] Attempting test trade: BUY 1 AAPL...")
    try:
        order = api.submit_order(
            symbol='AAPL',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"    Order submitted!")
        print(f"    Order ID: {order.id}")
        print(f"    Status: {order.status}")
        print(f"    Symbol: {order.symbol}")
        print(f"    Qty: {order.qty}")
        print(f"    Side: {order.side}")
    except Exception as e:
        print(f"    Order failed: {e}")

    print("\n" + "=" * 50)
    print("ALPACA CONNECTION: SUCCESS!")
    print("=" * 50)

except ImportError:
    print("\nERROR: alpaca-trade-api not installed")
    print("Run: pip install alpaca-trade-api")
except Exception as e:
    print(f"\nERROR: {e}")

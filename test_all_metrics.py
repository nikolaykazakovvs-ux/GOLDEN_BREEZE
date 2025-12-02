"""Comprehensive test of all 9 trading metrics."""
from mcp_servers.trading import metrics
from mcp_servers.trading.mt5_connector import get_connector
from datetime import datetime, timedelta

def test_all_metrics():
    print("=" * 70)
    print("Testing All 9 Trading Metrics")
    print("=" * 70)
    
    # Connect
    print("\n[1] Connecting to MT5...")
    connector = get_connector()
    if not connector.initialize():
        print("❌ Failed to connect")
        return
    print("✓ Connected")
    
    # Get metrics
    print("\n[2] Calculating metrics...")
    start = (datetime.now() - timedelta(days=30)).isoformat()
    overall = metrics.get_overall_metrics("current", start=start, timeframe="M5")
    
    # Test each metric
    print("\n[3] Testing each metric:\n")
    
    tests = [
        ("1. Date Start", overall.get('date_start'), lambda v: v is not None),
        ("2. Date End", overall.get('date_end'), lambda v: v is not None),
        ("3. ROI (%)", overall.get('roi_percent'), lambda v: isinstance(v, (int, float))),
        ("4. Net PnL ($)", overall.get('net_pnl'), lambda v: isinstance(v, (int, float))),
        ("5. Win Ratio (%)", overall.get('win_ratio_percent'), lambda v: 0 <= v <= 100),
        ("6. Max Drawdown (%)", overall.get('max_drawdown_percent'), lambda v: v >= 0),
        ("7. Time in Market (%)", overall.get('time_in_market_percent'), lambda v: v >= 0),
        ("8. Number of Trades", overall.get('number_of_trades'), lambda v: isinstance(v, int) and v >= 0),
        ("9. Average Trade Duration", overall.get('average_trade_duration'), lambda v: v is not None),
    ]
    
    tests.append(("Timeframe", overall.get('timeframe'), lambda v: v is not None))
    
    passed = 0
    failed = 0
    
    for name, value, validator in tests:
        try:
            if validator(value):
                print(f"  ✓ {name:30s} = {value}")
                passed += 1
            else:
                print(f"  ✗ {name:30s} = {value} (validation failed)")
                failed += 1
        except Exception as e:
            print(f"  ✗ {name:30s} ERROR: {e}")
            failed += 1
    
    # Additional checks
    print("\n[4] Additional validations:")
    
    # Equity curve
    equity = overall.get('equity_curve', [])
    if equity and len(equity) > 0:
        print(f"  ✓ Equity Curve: {len(equity)} points")
        passed += 1
    else:
        print(f"  ✗ Equity Curve: empty")
        failed += 1
    
    # Account info
    if 'account_info' in overall:
        print(f"  ✓ Account Info: present")
        passed += 1
    else:
        print(f"  ⚠ Account Info: missing (optional)")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    print("=" * 70)
    
    # Disconnect
    connector.shutdown()
    
    return failed == 0

if __name__ == "__main__":
    success = test_all_metrics()
    exit(0 if success else 1)

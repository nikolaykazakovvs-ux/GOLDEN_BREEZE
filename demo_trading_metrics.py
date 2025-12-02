"""Demo script: Trading Metrics with MT5 integration.

Demonstrates comprehensive metrics calculation including:
- Date Start/End, ROI, Net PnL, Win Ratio
- Max Drawdown, Time in Market, Number of Trades
- Average Trade Duration, Timeframe
"""
from mcp_servers.trading import metrics, trade_history
from mcp_servers.trading.mt5_connector import get_connector
from datetime import datetime, timedelta
import json

def print_metrics_report(metrics_data: dict):
    """Красиво форматировать и вывести метрики."""
    print("\n" + "=" * 70)
    print("📊 TRADING METRICS REPORT")
    print("=" * 70)
    
    # Период
    print(f"\n📅 Period:")
    print(f"  Date Start:  {metrics_data.get('date_start', 'N/A')}")
    print(f"  Date End:    {metrics_data.get('date_end', 'N/A')}")
    print(f"  Timeframe:   {metrics_data.get('timeframe', 'N/A')}")
    
    # Основные метрики
    print(f"\n💰 Performance:")
    net_pnl = metrics_data.get('net_pnl', 0)
    print(f"  Net PnL:     ${net_pnl:.2f} {'✓' if net_pnl > 0 else '✗'}")
    print(f"  ROI:         {metrics_data.get('roi_percent', 0):.2f}%")
    
    # Статистика сделок
    print(f"\n📈 Trade Statistics:")
    print(f"  Number of Trades:        {metrics_data.get('number_of_trades', 0)}")
    win_ratio = metrics_data.get('win_ratio_percent', 0)
    print(f"  Win Ratio:               {win_ratio:.2f}% {'✓' if win_ratio >= 50 else '⚠'}")
    print(f"  Average Trade Duration:  {metrics_data.get('average_trade_duration', 'N/A')}")
    
    # Риск-метрики
    print(f"\n⚠️  Risk Metrics:")
    max_dd = metrics_data.get('max_drawdown_percent', 0)
    print(f"  Max Drawdown:      {max_dd:.2f}% {'⚠' if max_dd > 20 else '✓'}")
    print(f"  Time in Market:    {metrics_data.get('time_in_market_percent', 0):.2f}%")
    
    # Информация об аккаунте
    if 'account_info' in metrics_data:
        acc = metrics_data['account_info']
        print(f"\n💼 Account Info:")
        print(f"  Login:          {acc.get('login', 'N/A')}")
        print(f"  Server:         {acc.get('server', 'N/A')}")
        print(f"  Current Balance: ${acc.get('current_balance', 0):.2f}")
        print(f"  Current Equity:  ${acc.get('current_equity', 0):.2f}")
    
    print("\n" + "=" * 70)

def main():
    print("=" * 70)
    print("Golden Breeze Trading Metrics Demo")
    print("=" * 70)
    
    # 1. Подключение к MT5
    print("\n[1] Connecting to MT5...")
    connector = get_connector()
    
    if not connector.initialize():
        print("❌ Failed to connect to MT5")
        return
    
    print("✓ Connected to MT5")
    
    # 2. Получение информации об аккаунте
    account_info = connector.get_account_info()
    if account_info:
        print(f"  Account: {account_info['login']} @ {account_info['server']}")
        print(f"  Balance: {account_info['balance']} {account_info['currency']}")
    
    # 3. Полный набор метрик за последние 30 дней
    print("\n[2] Calculating metrics for last 30 days...")
    start = (datetime.now() - timedelta(days=30)).isoformat()
    
    overall = metrics.get_overall_metrics(
        account_id="current",
        start=start,
        timeframe="M5"
    )
    
    print_metrics_report(overall)
    
    # 4. Equity Curve
    print("\n[3] Building equity curve...")
    equity = metrics.get_equity_curve("current", start=start)
    
    if equity:
        print(f"✓ Equity curve: {len(equity)} points")
        print(f"  Start Balance: ${equity[0]:.2f}")
        print(f"  End Balance:   ${equity[-1]:.2f}")
        print(f"  Peak Balance:  ${max(equity):.2f}")
        print(f"  Low Balance:   ${min(equity):.2f}")
    else:
        print("  No equity data available")
    
    # 5. Статистика по символу XAUUSD
    print("\n[4] Symbol statistics (XAUUSD)...")
    regime_stats = metrics.get_regime_stats(
        account_id="current",
        symbol="XAUUSD",
        regime="All",
        start=start
    )
    
    print(f"  Symbol:       {regime_stats.get('symbol', 'N/A')}")
    print(f"  Trades:       {regime_stats.get('trades_count', 0)}")
    print(f"  Win Ratio:    {regime_stats.get('win_ratio', 0):.2f}%")
    print(f"  Net PnL:      ${regime_stats.get('net_pnl', 0):.2f}")
    print(f"  Avg Profit:   ${regime_stats.get('avg_profit', 0):.2f}")
    
    # 6. Метрики за последние 7 дней (для сравнения)
    print("\n[5] Comparing with last 7 days...")
    start_7d = (datetime.now() - timedelta(days=7)).isoformat()
    
    metrics_7d = metrics.get_overall_metrics(
        account_id="current",
        start=start_7d,
        timeframe="M5"
    )
    
    print(f"  Last 7 days:")
    print(f"    Trades:      {metrics_7d.get('number_of_trades', 0)}")
    print(f"    Net PnL:     ${metrics_7d.get('net_pnl', 0):.2f}")
    print(f"    Win Ratio:   {metrics_7d.get('win_ratio_percent', 0):.2f}%")
    print(f"    ROI:         {metrics_7d.get('roi_percent', 0):.2f}%")
    
    # 7. Сохранение метрик в JSON
    print("\n[6] Saving metrics to file...")
    metrics_file = "metrics_report.json"
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "period_30d": overall,
        "period_7d": metrics_7d,
        "equity_curve": equity if equity else [],
        "symbol_stats": regime_stats,
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metrics saved to {metrics_file}")
    
    # Отключение
    print("\n[7] Disconnecting from MT5...")
    connector.shutdown()
    print("✓ Disconnected")
    
    print("\n" + "=" * 70)
    print("Metrics calculation completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""Visualization of trading metrics and equity curve.

Requires matplotlib. Install: pip install matplotlib
"""
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed. Run: pip install matplotlib")

from mcp_servers.trading import metrics, trade_history
from mcp_servers.trading.mt5_connector import get_connector
from datetime import datetime, timedelta
import json

def plot_equity_curve(equity: list, trades: list, title: str = "Equity Curve"):
    """Построить график equity curve с отметками сделок."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # График equity
    ax1.plot(range(len(equity)), equity, linewidth=2, color='#2E86DE', label='Equity')
    ax1.fill_between(range(len(equity)), equity, alpha=0.3, color='#2E86DE')
    ax1.axhline(y=equity[0], color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
    
    # Отметки выигрышных и проигрышных сделок
    wins = [i for i, t in enumerate(trades) if t['profit'] > 0]
    losses = [i for i, t in enumerate(trades) if t['profit'] <= 0]
    
    if wins:
        ax1.scatter([w + 1 for w in wins], [equity[w + 1] for w in wins], 
                   color='green', marker='^', s=100, zorder=5, label='Win', alpha=0.7)
    if losses:
        ax1.scatter([l + 1 for l in losses], [equity[l + 1] for l in losses],
                   color='red', marker='v', s=100, zorder=5, label='Loss', alpha=0.7)
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Trade Number', fontsize=12)
    ax1.set_ylabel('Balance ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Drawdown график
    peak = equity[0]
    drawdowns = []
    for val in equity:
        if val > peak:
            peak = val
        dd = ((peak - val) / peak) * 100 if peak > 0 else 0
        drawdowns.append(-dd)  # Отрицательные значения для визуализации
    
    ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.5)
    ax2.plot(range(len(drawdowns)), drawdowns, color='darkred', linewidth=2)
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
    print("✓ Chart saved to equity_curve.png")
    plt.show()

def plot_metrics_summary(metrics_data: dict):
    """Построить сводную диаграмму метрик."""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not installed")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROI и Net PnL
    roi = metrics_data.get('roi_percent', 0)
    pnl = metrics_data.get('net_pnl', 0)
    
    ax1.bar(['ROI (%)', 'Net PnL ($)'], [roi, pnl], color=['#2E86DE', '#27AE60' if pnl > 0 else '#E74C3C'])
    ax1.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([roi, pnl]):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=12, fontweight='bold')
    
    # 2. Win Ratio
    win_ratio = metrics_data.get('win_ratio_percent', 0)
    loss_ratio = 100 - win_ratio
    
    colors = ['#27AE60', '#E74C3C']
    ax2.pie([win_ratio, loss_ratio], labels=['Wins', 'Losses'], autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Win/Loss Ratio', fontsize=14, fontweight='bold')
    
    # 3. Max Drawdown
    max_dd = metrics_data.get('max_drawdown_percent', 0)
    
    ax3.barh(['Max Drawdown'], [max_dd], color='#E74C3C', alpha=0.7)
    ax3.set_xlim(0, max(max_dd * 1.2, 10))
    ax3.set_title('Risk Metrics', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.text(max_dd, 0, f'{max_dd:.2f}%', va='center', ha='left', fontsize=12, fontweight='bold')
    
    # 4. Trade Statistics
    num_trades = metrics_data.get('number_of_trades', 0)
    time_in_market = metrics_data.get('time_in_market_percent', 0)
    
    ax4.bar(['Trades', 'Time in Market (%)'], [num_trades, time_in_market], color=['#9B59B6', '#F39C12'])
    ax4.set_title('Activity Metrics', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([num_trades, time_in_market]):
        ax4.text(i, v, f'{v:.0f}' if i == 0 else f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Chart saved to metrics_summary.png")
    plt.show()

def main():
    print("=" * 70)
    print("Golden Breeze Trading Metrics Visualization")
    print("=" * 70)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  matplotlib not installed")
        print("Install it with: pip install matplotlib")
        return
    
    # Подключение к MT5
    print("\n[1] Connecting to MT5...")
    connector = get_connector()
    if not connector.initialize():
        print("❌ Failed to connect to MT5")
        return
    print("✓ Connected")
    
    # Получение метрик
    print("\n[2] Calculating metrics...")
    start = (datetime.now() - timedelta(days=30)).isoformat()
    
    overall = metrics.get_overall_metrics("current", start=start, timeframe="M5")
    equity = metrics.get_equity_curve("current", start=start)
    trades = trade_history.get_closed_trades("current", start=start)
    
    print(f"✓ Metrics calculated")
    print(f"  Trades: {len(trades)}")
    print(f"  Net PnL: ${overall.get('net_pnl', 0):.2f}")
    print(f"  Win Ratio: {overall.get('win_ratio_percent', 0):.2f}%")
    
    # Построение графиков
    print("\n[3] Building charts...")
    
    if equity and trades:
        timeframe: str = str(overall.get('timeframe', 'M5'))
        date_start: str = str(overall.get('date_start', 'N/A'))[:10]
        date_end: str = str(overall.get('date_end', 'N/A'))[:10]
        
        plot_equity_curve(equity, trades, 
                         title=f"Equity Curve - {timeframe} ({date_start} to {date_end})")
    
    plot_metrics_summary(overall)
    
    print("\n✓ Visualization complete!")
    print("  Charts saved: equity_curve.png, metrics_summary.png")
    
    # Отключение
    connector.shutdown()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

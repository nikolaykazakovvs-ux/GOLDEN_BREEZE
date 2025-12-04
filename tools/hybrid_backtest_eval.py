"""tools.hybrid_backtest_eval

Запуск полноценного backtest для Golden Breeze Hybrid Strategy v1.1
с использованием обученной Direction LSTM v1.1 и сбором метрик/отчётов.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy import HybridStrategy, StrategyConfig
from strategy.backtest_engine import BacktestEngine

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, ATR, RSI indicators mirroring demo_backtest_hybrid."""
    data = df.copy()
    data["sma_fast"] = data["close"].rolling(window=20).mean()
    data["sma_slow"] = data["close"].rolling(window=50).mean()

    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close = (data["low"] - data["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(window=14).mean()

    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    data = data.fillna(method="ffill").fillna(method="bfill")
    return data



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Golden Breeze Hybrid Strategy backtest & evaluation"
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--primary-tf", default="M5", dest="primary_tf")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--ai-url", default="http://127.0.0.1:5005")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--use-tick-data", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--output-tag", default="v1.1")
    return parser.parse_args()


def load_csv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = DATA_ROOT / symbol / f"{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    time_column = None
    for candidate in ("time", "timestamp", "datetime"):
        if candidate in df.columns:
            time_column = candidate
            break
    if time_column is None:
        raise ValueError(f"CSV {path} не содержит колонку времени")
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df.set_index(time_column, inplace=True)
    # Удаляем потенциальный пустой индекс
    df = df.sort_index()
    columns = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    return df[columns]


def slice_period(df: pd.DataFrame, start: Optional[str], end: Optional[str], lookback_days: int) -> Tuple[pd.DataFrame, datetime, datetime]:
    if df is None or df.empty:
        raise ValueError("Пустой DataFrame для выбора периода")
    if start and end:
        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
    else:
        end_ts = df.index.max()
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
    sliced = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if sliced.empty:
        raise ValueError("Нет данных в выбранном диапазоне")
    return sliced, start_ts.to_pydatetime(), end_ts.to_pydatetime()


def ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)


def fetch_health(ai_url: str) -> Dict:
    resp = requests.get(f"{ai_url.rstrip('/')}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def validate_health(health: Dict) -> None:
    if str(health.get("device")) != "cuda" or not health.get("use_gpu"):
        raise RuntimeError("AI Core не использует CUDA. Запустите ядро на GPU перед backtest.")
    if "direction_model" not in health:
        raise RuntimeError("/health не возвращает информацию о модели направления")


def build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    config = StrategyConfig(
        symbol=args.symbol,
        primary_tf=args.primary_tf,
        base_timeframe=args.primary_tf,
        risk_per_trade_pct=args.risk_pct,
        ai_api_url=args.ai_url,
        min_direction_confidence=args.min_confidence,
        use_tick_data=args.use_tick_data,
        initial_balance=args.initial_balance,
    )
    return config


def compute_time_in_market(trades: List) -> Tuple[float, float]:
    if not trades:
        return 0.0, 0.0
    total_seconds = 0.0
    for trade in trades:
        if trade.entry_time and trade.exit_time:
            total_seconds += max(0.0, (trade.exit_time - trade.entry_time).total_seconds())
    total_hours = total_seconds / 3600.0
    return total_seconds, total_hours


def compute_metrics(strategy: HybridStrategy, equity_df: pd.DataFrame, period_start: datetime, period_end: datetime) -> Dict:
    stats = strategy.get_statistics()
    initial_balance = strategy.risk_manager.initial_balance
    final_balance = stats["current_balance"]
    total_pnl = stats["total_pnl"]
    roi_pct = (total_pnl / initial_balance * 100.0) if initial_balance else 0.0

    trades = strategy.risk_manager.trade_history
    gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
    gross_loss = sum(t.pnl for t in trades if t.pnl and t.pnl < 0)
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf") if gross_profit > 0 else 0.0

    eq_curve = equity_df.copy()
    eq_curve["timestamp"] = pd.to_datetime(eq_curve["timestamp"], utc=True)
    eq_curve.sort_values("timestamp", inplace=True)

    max_dd_pct = 0.0
    max_dd_abs = 0.0
    if not eq_curve.empty:
        balance = eq_curve["balance"].astype(float)
        running_max = balance.cummax()
        drawdowns = (balance - running_max)
        drawdown_pct = (drawdowns / running_max.replace(0, pd.NA)) * 100
        max_dd_abs = drawdowns.min() if not drawdowns.empty else 0.0
        max_dd_pct = drawdown_pct.min() if not drawdown_pct.empty else 0.0
    max_dd_pct = float(abs(max_dd_pct))
    max_dd_abs = float(abs(max_dd_abs))

    sharpe_ratio = 0.0
    if not eq_curve.empty:
        daily_balance = eq_curve.set_index("timestamp")["balance"].resample("1D").last().dropna()
        returns = daily_balance.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = float((returns.mean() / returns.std()) * math.sqrt(252))

    avg_trade_pnl = stats["avg_pnl"] if stats["total_trades"] > 0 else 0.0
    total_seconds, total_hours = compute_time_in_market(trades)
    total_period_seconds = max(1.0, (period_end - period_start).total_seconds())
    time_in_market_pct = (total_seconds / total_period_seconds) * 100.0

    long_trades = len([t for t in trades if t.direction == "long"])
    short_trades = len([t for t in trades if t.direction == "short"])

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "net_pnl": total_pnl,
        "roi_pct": roi_pct,
        "win_rate_pct": stats["win_rate"],
        "total_trades": stats["total_trades"],
        "avg_trade_pnl": avg_trade_pnl,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_abs": max_dd_abs,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "time_in_market_pct": time_in_market_pct,
        "time_in_market_hours": total_hours,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "regime_stats": strategy.risk_manager.get_regime_statistics(),
    }


def format_equity_notes(metrics: Dict) -> str:
    if metrics["roi_pct"] >= 0:
        note = (
            f"Equity росла с ${metrics['initial_balance']:,.0f} до ${metrics['final_balance']:,.0f}, "
            f"совокупный результат {metrics['roi_pct']:.2f}% при max DD {metrics['max_drawdown_pct']:.2f}%."
        )
    else:
        note = (
            f"Equity снизилась до ${metrics['final_balance']:,.0f} (ROI {metrics['roi_pct']:.2f}%), "
            f"просадка достигала {metrics['max_drawdown_pct']:.2f}% и требует оптимизации."
        )
    return note


def auto_observations(metrics: Dict) -> List[str]:
    notes = []
    if metrics["roi_pct"] > 0:
        notes.append(f"Стратегия заработала {metrics['roi_pct']:.2f}% за период, что подтверждает потенциал модели.")
    else:
        notes.append("ROI оказался отрицательным — требуется дополнительный тюнинг торговых фильтров.")

    if metrics["profit_factor"] < 1:
        notes.append("Profit Factor < 1, потери преобладают — нужно улучшить фильтрацию входов и управление риском.")
    else:
        notes.append(f"Profit Factor {metrics['profit_factor']:.2f}, что говорит о положительном соотношении прибыль/убыток.")

    if metrics["max_drawdown_pct"] > 10:
        notes.append("Максимальная просадка превышает 10% — стоит ужесточить лимиты или сократить риск на сделку.")
    else:
        notes.append("Просадка удержана в разумных пределах (<10%), риск-менеджмент работает стабильно.")

    if metrics["long_trades"] > metrics["short_trades"] * 1.5:
        notes.append("Наблюдается перекос в пользу LONG-сделок — стоит проверить качество сигналов SHORT.")
    elif metrics["short_trades"] > metrics["long_trades"] * 1.5:
        notes.append("Стратегия чаще шортит рынок — убедитесь, что bearish сценарии оправданы.")
    else:
        notes.append("Распределение LONG/SHORT сбалансировано, стратегия использует оба направления.")
    return notes[:4]


def default_next_steps(metrics: Dict) -> List[str]:
    steps = [
        "Подобрать более высокий порог confidence для входов на флэтовых участках.",
        "Добавить фильтр по волатильности (ATR) перед открытием сделок.",
        "Проанализировать режимы с худшей статистикой и откорректировать правила выхода.",
    ]
    if metrics["profit_factor"] < 1:
        steps.append("Оптимизировать риск-менеджмент (SL/TP) для улучшения Profit Factor.")
    return steps


def save_json(metrics: Dict, meta: Dict, period: Tuple[datetime, datetime], filename: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {
            "start": period[0].isoformat(),
            "end": period[1].isoformat(),
            "days": (period[1] - period[0]).days,
        },
        "model": meta,
        "metrics": metrics,
    }
    filename.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def save_markdown(metrics: Dict, meta: Dict, period: Tuple[datetime, datetime], health: Dict,
                  summary_note: str, observations: List[str], next_steps: List[str],
                  regime_stats: Dict, filename: Path) -> None:
    start_str = period[0].strftime("%Y-%m-%d %H:%M")
    end_str = period[1].strftime("%Y-%m-%d %H:%M")
    md_lines = [
        "# HYBRID_BACKTEST_EVAL_v1.1",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Environment",
        "",
        f"- **device:** {health.get('device')}",
        f"- **use_gpu:** {health.get('use_gpu')}",
        f"- **torch_version:** {health.get('torch_version', 'n/a')}",
        "- **model:** {meta.get('name')}",
        f"- **seq_len:** {meta.get('seq_len')}",
        f"- **test_accuracy:** {meta.get('test_accuracy')}",
        f"- **test_f1_macro:** {meta.get('test_f1_macro')}",
        f"- **test_mcc:** {meta.get('test_mcc')}",
        f"- **period:** {start_str} → {end_str}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| ROI % | {metrics['roi_pct']:.2f}% |",
        f"| Net PnL | ${metrics['net_pnl']:,.2f} |",
        f"| Win Rate | {metrics['win_rate_pct']:.2f}% |",
        f"| Profit Factor | {metrics['profit_factor']:.2f} |",
        f"| Max Drawdown | {metrics['max_drawdown_pct']:.2f}% (${metrics['max_drawdown_abs']:,.2f}) |",
        f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} |",
        f"| Trades | {metrics['total_trades']} |",
        f"| Avg Trade PnL | ${metrics['avg_trade_pnl']:,.2f} |",
        f"| Time In Market | {metrics['time_in_market_pct']:.2f}% ({metrics['time_in_market_hours']:.1f}h) |",
        "",
        "## Equity & Drawdown",
        "",
        summary_note,
        "",
        "## Regime Analysis",
        "",
    ]
    if regime_stats:
        md_lines.append("| Regime | Trades | Win Rate | PnL |")
        md_lines.append("|--------|--------|----------|-----|")
        for regime, stats in regime_stats.items():
            md_lines.append(
                f"| {regime} | {stats['trades']} | {stats['win_rate']:.1f}% | ${stats['total_pnl']:,.2f} |"
            )
        md_lines.append("")
    else:
        md_lines.append("Данные по режимам недоступны.\n")

    md_lines.append("## Observations")
    md_lines.append("")
    for obs in observations:
        md_lines.append(f"- {obs}")
    md_lines.append("")
    md_lines.append("## Next Steps")
    md_lines.append("")
    for step in next_steps:
        md_lines.append(f"1. {step}")
    md_lines.append("")
    filename.write_text("\n".join(md_lines), encoding="utf-8")


def main():
    args = parse_args()
    ensure_reports_dir()

    health = fetch_health(args.ai_url)
    validate_health(health)
    model_meta = health.get("direction_model", {})

    m5 = load_csv(args.symbol, "M5")
    if m5 is None or m5.empty:
        raise FileNotFoundError("Не удалось загрузить M5 данные из data/raw")
    m5, start_dt, end_dt = slice_period(m5, args.start, args.end, args.lookback_days)
    m5 = prepare_data_with_indicators(m5)

    data_dict = {"M5": m5}
    for tf in ["M15", "H1", "H4"]:
        df = load_csv(args.symbol, tf)
        if df is not None and not df.empty:
            df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
            if len(df) > 0:
                data_dict[tf] = prepare_data_with_indicators(df)

    m1 = load_csv(args.symbol, "M1")
    if m1 is not None and not m1.empty:
        m1 = m1[(m1.index >= start_dt) & (m1.index <= end_dt)].copy()

    config = build_strategy_config(args)
    strategy = HybridStrategy(config, initial_balance=args.initial_balance)
    engine = BacktestEngine(strategy, config)
    engine.load_multitf_data(data_dict)
    if m1 is not None and not m1.empty:
        engine.load_m1_data(m1)

    engine.run(start_date=start_dt, end_date=end_dt)
    equity_df = engine.get_equity_curve()
    metrics = compute_metrics(strategy, equity_df, start_dt, end_dt)
    summary_note = format_equity_notes(metrics)
    observations = auto_observations(metrics)
    next_steps = default_next_steps(metrics)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"hybrid_backtest_stats_{args.symbol}_{timestamp}.json"
    md_path = REPORTS_DIR / f"HYBRID_BACKTEST_EVAL_v1.1_{args.symbol}_{timestamp}.md"

    save_json(metrics, model_meta, (start_dt, end_dt), json_path)
    save_markdown(metrics, model_meta, (start_dt, end_dt), health, summary_note, observations, next_steps,
                  metrics.get("regime_stats", {}), md_path)

    print("============================================================")
    print("✅ HYBRID_BACKTEST_EVAL_v1.1 completed")
    print(f"JSON metrics: {json_path}")
    print(f"Markdown report: {md_path}")
    print("============================================================")


if __name__ == "__main__":
    main()

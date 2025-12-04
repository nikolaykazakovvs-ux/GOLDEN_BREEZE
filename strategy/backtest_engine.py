# strategy/backtest_engine.py
"""
Backtesting engine с поддержкой тиков и M1 данных
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .hybrid_strategy import HybridStrategy
from .intrabar_engine import Tick, IntrabarCandle
from .config import StrategyConfig


class BacktestEngine:
    """
    Движок для backtesting с интрабарной логикой и мультитаймфреймом.
    
    Поддерживает:
    - Мультитаймфреймовые данные (M5, M15, H1, H4)
    - Тиковые данные MT5
    - M1 данные для интрабарной симуляции
    """
    
    def __init__(self, strategy: HybridStrategy, config: StrategyConfig):
        self.strategy = strategy
        self.config = config
        
        # Данные по таймфреймам
        self.multitf_data: Dict[str, pd.DataFrame] = {}  # {tf: DataFrame}
        self.m5_data: Optional[pd.DataFrame] = None  # Основной TF (для обратной совместимости)
        self.m1_data: Optional[pd.DataFrame] = None  # Интрабар
        self.tick_data: Optional[pd.DataFrame] = None  # Тики
        
        # Результаты
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = []
    
    def load_multitf_data(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Загрузка данных по нескольким таймфреймам.
        
        Args:
            data_dict: {tf: DataFrame} например {"M5": df_m5, "M15": df_m15, "H1": df_h1, "H4": df_h4}
        """
        for tf, data in data_dict.items():
            df = data.copy()
            # FIX: Правильная обработка timestamp
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.dropna(subset=['time'])
                df.set_index('time', inplace=True, drop=False)
            else:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]
            
            self.multitf_data[tf] = df
            
            # Для обратной совместимости
            if tf == "M5" or (self.m5_data is None and tf == self.config.primary_tf):
                self.m5_data = df
        
        print(f"Loaded multitimeframe data: {list(self.multitf_data.keys())}")
    
    def load_m5_data(self, data: pd.DataFrame):
        """Загрузка M5 данных с индикаторами (обратная совместимость)"""
        self.m5_data = data.copy()
        # FIX: Правильная обработка timestamp
        if 'time' in self.m5_data.columns:
            self.m5_data['time'] = pd.to_datetime(self.m5_data['time'], errors='coerce')
            self.m5_data = self.m5_data.dropna(subset=['time'])
            self.m5_data.set_index('time', inplace=True, drop=False)
        else:
            self.m5_data.index = pd.to_datetime(self.m5_data.index, errors='coerce')
            self.m5_data = self.m5_data[self.m5_data.index.notna()]
        self.multitf_data["M5"] = self.m5_data
    
    def load_m1_data(self, data: pd.DataFrame):
        """Загрузка M1 данных для интрабарной симуляции"""
        self.m1_data = data.copy()
        # FIX: Правильная обработка timestamp
        if 'time' in self.m1_data.columns:
            self.m1_data['time'] = pd.to_datetime(self.m1_data['time'], errors='coerce')
            self.m1_data = self.m1_data.dropna(subset=['time'])
            self.m1_data.set_index('time', inplace=True, drop=False)
        else:
            self.m1_data.index = pd.to_datetime(self.m1_data.index, errors='coerce')
            self.m1_data = self.m1_data[self.m1_data.index.notna()]
    
    def load_tick_data(self, data: pd.DataFrame):
        """Загрузка тиковых данных"""
        self.tick_data = data.copy()
        # FIX: Правильная обработка timestamp
        if 'time' in self.tick_data.columns:
            self.tick_data['time'] = pd.to_datetime(self.tick_data['time'], errors='coerce')
            self.tick_data = self.tick_data.dropna(subset=['time'])
            self.tick_data.set_index('time', inplace=True, drop=False)
        else:
            self.tick_data.index = pd.to_datetime(self.tick_data.index, errors='coerce')
            self.tick_data = self.tick_data[self.tick_data.index.notna()]
    
    def load_csv_data(self, filepath: str, timeframe: str = "M5"):
        """
        Загрузка данных из CSV с правильным парсингом дат.
        
        Args:
            filepath: Путь к CSV файлу
            timeframe: Таймфрейм данных (M5, M15, H1, H4)
        """
        # FIX: Используем parse_dates для правильной загрузки
        df = pd.read_csv(filepath, parse_dates=['time'])
        
        # Конвертируем time в datetime
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df.set_index('time', inplace=True, drop=False)
        
        # Загружаем в соответствующий таймфрейм
        self.multitf_data[timeframe] = df
        if timeframe == "M5":
            self.m5_data = df
        
        print(f"✅ Loaded {len(df)} candles from {filepath} ({timeframe})")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Запуск backtesting
        
        Args:
            start_date: Дата начала (YYYY-MM-DD)
            end_date: Дата окончания (YYYY-MM-DD)
        """
        if self.m5_data is None:
            raise ValueError("M5 data not loaded")
        
        # Фильтрация по датам
        data = self.m5_data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        print(f"\n{'='*60}")
        print(f"Golden Breeze Hybrid Strategy - Backtest")
        print(f"{'='*60}")
        print(f"Period: {data.index[0]} to {data.index[-1]}")
        print(f"Bars: {len(data)}")
        print(f"Initial Balance: ${self.strategy.risk_manager.initial_balance:,.2f}")
        print(f"{'='*60}\n")
        
        # Основной цикл по M5 свечам
        for i in range(50, len(data)):  # Начинаем с 50 для индикаторов
            current_bar = data.iloc[i]
            historical_data = data.iloc[:i+1]
            
            # Формирование свечи
            candle = {
                "timestamp": str(current_bar.name),
                "open": current_bar["open"],
                "high": current_bar["high"],
                "low": current_bar["low"],
                "close": current_bar["close"],
                "volume": current_bar.get("volume", 0)
            }
            
            # Обработка M5 свечи (передаём мультитаймфреймовые данные)
            # Синхронизируем данные по всем таймфреймам на момент текущей M5 свечи
            synced_multitf_data = self._sync_multitf_data(current_bar.name, i)
            
            self.strategy.on_new_candle(candle, historical_data, synced_multitf_data)
            
            # Интрабарная обработка
            if self.config.use_tick_data and self.tick_data is not None:
                # Используем реальные тики
                self._process_ticks_for_bar(current_bar.name)
            elif self.m1_data is not None:
                # Используем M1 для симуляции
                self._process_m1_for_bar(current_bar.name)
            else:
                # Простая симуляция: Open -> High -> Low -> Close
                self._simple_intrabar_simulation(current_bar)
            
            # Запись equity
            stats = self.strategy.get_statistics()
            self.equity_curve.append({
                "timestamp": current_bar.name,
                "balance": stats["current_balance"],
                "open_positions": stats["open_positions"]
            })
            
            # Прогресс
            if i % 100 == 0:
                pct = (i / len(data)) * 100
                print(f"Progress: {pct:.1f}% | Balance: ${stats['current_balance']:,.2f} | "
                      f"Trades: {stats['total_trades']} | DD: {stats['current_dd_pct']:.2f}%")
        
        # Финальная статистика
        self._print_results()
    
    def _process_ticks_for_bar(self, bar_time: pd.Timestamp):
        """Обработка реальных тиков для M5 бара"""
        if self.tick_data is None:
            return
        
        # Тики для текущего M5 бара (следующие 5 минут)
        end_time = bar_time + timedelta(minutes=5)
        ticks = self.tick_data[(self.tick_data.index >= bar_time) & 
                               (self.tick_data.index < end_time)]
        
        for idx, tick_row in ticks.iterrows():
            tick = Tick(
                timestamp=idx,
                bid=tick_row["bid"],
                ask=tick_row["ask"],
                volume=tick_row.get("volume", 0)
            )
            self.strategy.on_tick(tick)
    
    def _process_m1_for_bar(self, bar_time: pd.Timestamp):
        """Обработка M1 свечей для интрабарной симуляции"""
        if self.m1_data is None:
            return
        
        # M1 свечи для текущего M5 бара
        end_time = bar_time + timedelta(minutes=5)
        m1_bars = self.m1_data[(self.m1_data.index >= bar_time) & 
                               (self.m1_data.index < end_time)]
        
        for idx, m1_row in m1_bars.iterrows():
            m1_candle = IntrabarCandle(
                timestamp=idx,
                open=m1_row["open"],
                high=m1_row["high"],
                low=m1_row["low"],
                close=m1_row["close"],
                volume=m1_row.get("volume", 0)
            )
            self.strategy.on_m1_candle(m1_candle)
    
    def _simple_intrabar_simulation(self, bar: pd.Series):
        """
        Простая интрабарная симуляция без M1/тиков
        
        Последовательность: Open -> High -> Low -> Close
        """
        spread = 0.5  # Примерный спред
        
        # 1. Open
        tick = Tick(
            timestamp=bar.name,
            bid=bar["open"] - spread/2,
            ask=bar["open"] + spread/2
        )
        self.strategy.on_tick(tick)
        
        # 2. High
        tick = Tick(
            timestamp=bar.name,
            bid=bar["high"] - spread/2,
            ask=bar["high"] + spread/2
        )
        self.strategy.on_tick(tick)
        
        # 3. Low
        tick = Tick(
            timestamp=bar.name,
            bid=bar["low"] - spread/2,
            ask=bar["low"] + spread/2
        )
        self.strategy.on_tick(tick)
        
        # 4. Close
        tick = Tick(
            timestamp=bar.name,
            bid=bar["close"] - spread/2,
            ask=bar["close"] + spread/2
        )
        self.strategy.on_tick(tick)
    
    def _print_results(self):
        """Вывод результатов backtesting"""
        stats = self.strategy.get_statistics()
        regime_stats = stats["regime_stats"]
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Overall Performance:")
        print(f"  Initial Balance:  ${self.strategy.risk_manager.initial_balance:,.2f}")
        print(f"  Final Balance:    ${stats['current_balance']:,.2f}")
        print(f"  Net PnL:          ${stats['total_pnl']:,.2f}")
        print(f"  ROI:              {(stats['total_pnl'] / self.strategy.risk_manager.initial_balance) * 100:.2f}%")
        print(f"  Max Drawdown:     {stats['current_dd_pct']:.2f}%")
        
        print(f"\n📈 Trading Statistics:")
        print(f"  Total Trades:     {stats['total_trades']}")
        print(f"  Wins:             {stats['wins']}")
        print(f"  Losses:           {stats['losses']}")
        print(f"  Win Rate:         {stats['win_rate']:.2f}%")
        print(f"  Avg PnL:          ${stats['avg_pnl']:.2f}")
        
        print(f"\n🎯 Performance by Regime:")
        for regime, regime_stat in regime_stats.items():
            print(f"  {regime:12s}: {regime_stat['trades']:3d} trades, "
                  f"Win Rate: {regime_stat['win_rate']:5.1f}%, "
                  f"PnL: ${regime_stat['total_pnl']:8,.2f}")
        
        print(f"\n{'='*60}\n")
    
    def _sync_multitf_data(self, current_timestamp, current_index: int) -> Dict[str, pd.DataFrame]:
        """
        Синхронизирует данные по всем таймфреймам на момент текущей свечи.
        
        Args:
            current_timestamp: Временная метка текущей M5 свечи
            current_index: Индекс текущей свечи в M5 данных
        
        Returns:
            {tf: DataFrame} с данными по каждому таймфрейму до текущего момента
        """
        synced_data = {}
        
        for tf, data in self.multitf_data.items():
            # Фильтруем данные до текущего времени (включительно)
            mask = data.index <= current_timestamp
            synced_data[tf] = data[mask].copy()
        
        return synced_data
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Получение equity curve"""
        return pd.DataFrame(self.equity_curve)
    
    def export_results(self, filename: str = "backtest_results.csv"):
        """Экспорт результатов в CSV"""
        trades_df = pd.DataFrame([
            {
                "id": t.id,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "volume": t.volume,
                "pnl": t.pnl,
                "regime": t.regime,
                "reason": t.reason
            }
            for t in self.strategy.risk_manager.trade_history
        ])
        
        trades_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

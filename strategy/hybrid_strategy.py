# strategy/hybrid_strategy.py
"""
Golden Breeze Hybrid Strategy - главный класс стратегии
"""

from typing import Dict, Optional, List
from datetime import datetime, date
import pandas as pd
import numpy as np

from .config import StrategyConfig
from .intrabar_engine import IntrabarEngine, Tick, IntrabarCandle
from .regime_strategies import TrendStrategy, RangeStrategy, VolatileStrategy
from .risk_manager import RiskManager, Trade
from .ai_client import AIClient
from .timeframe_selector import TimeframeSelector, TimeframeData, Timeframe, Regime, TimeframeDecision


class HybridStrategy:
    """
    Гибридная торговая стратегия с интрабарной логикой и мультитаймфреймом.
    
    Объединяет:
    - AI сигналы по всем таймфреймам (M5, M15, H1, H4)
    - Динамический выбор PRIMARY_TF через TimeframeSelector
    - Режимные стратегии (trend/range/volatile)
    - Интрабарную обработку (тики/M1)
    - Строгий risk management
    """
    
    def __init__(self, config: StrategyConfig, initial_balance: float = 10000.0):
        self.config = config
        config.validate()
        
        # Компоненты
        self.ai_client = AIClient(config.ai_api_url)
        self.intrabar_engine = IntrabarEngine(config)
        self.risk_manager = RiskManager(config, initial_balance)
        
        # Мультитаймфреймовый селектор
        if config.tf_selector_enable:
            self.tf_selector = TimeframeSelector(
                default_primary_tf=Timeframe(config.primary_tf),
                min_confidence_threshold=config.tf_selector_min_confidence,
                high_confidence_threshold=config.tf_selector_high_confidence,
                context_tf_high=Timeframe(config.context_tf_high),
                exec_tf_low=Timeframe(config.execution_tf)
            )
        else:
            self.tf_selector = None
        
        # Режимные стратегии
        self.strategies = {
            "trend_up": TrendStrategy(config),
            "trend_down": TrendStrategy(config),
            "range": RangeStrategy(config),
            "volatile": VolatileStrategy(config)
        }
        
        # Текущее состояние
        self.current_regime = "unknown"
        self.current_primary_tf = config.primary_tf  # Динамически изменяемый
        self.current_ai_signal: Optional[Dict] = None
        self.current_multitf_signals: Optional[Dict[str, Dict]] = None  # {tf: signal}
        self.current_tf_decision: Optional[TimeframeDecision] = None
        self.pending_orders: Dict[str, Dict] = {}
        
        # Данные по всем таймфреймам
        self.multitf_data: Dict[str, pd.DataFrame] = {}  # {tf: DataFrame}
        self.data: Optional[pd.DataFrame] = None  # PRIMARY_TF data
        self.current_date: Optional[date] = None
    
    def on_new_candle(
        self, 
        candle: Dict, 
        historical_data: pd.DataFrame,
        multitf_data: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """
        Обработка новой свечи с мультитаймфреймовой логикой.
        
        Args:
            candle: Новая свеча {timestamp, open, high, low, close, volume}
            historical_data: Исторические данные с индикаторами (PRIMARY_TF)
            multitf_data: Данные по всем таймфреймам {tf: DataFrame}
        """
        timestamp = pd.to_datetime(candle["timestamp"])
        
        # Обновление текущей даты
        if self.current_date != timestamp.date():
            self.current_date = timestamp.date()
            self.risk_manager.reset_daily_limits()
        
        # Сохранение данных
        self.data = historical_data
        if multitf_data:
            self.multitf_data = multitf_data
        
        # ШАГ 1: Запрос AI сигналов по всем таймфреймам
        multitf_signals = self._request_multitf_signals()
        if not multitf_signals:
            return
        
        self.current_multitf_signals = multitf_signals
        
        # ШАГ 2: Выбор PRIMARY_TF через TimeframeSelector
        if self.tf_selector:
            tf_decision = self._select_primary_timeframe(multitf_signals)
            self.current_tf_decision = tf_decision
            self.current_primary_tf = tf_decision.primary_tf.value
            
            # Проверяем, можно ли торговать
            if not tf_decision.should_trade:
                print(f"[{timestamp}] TimeframeSelector: NO TRADE - {tf_decision.reason}")
                return
            
            print(f"[{timestamp}] PRIMARY_TF: {self.current_primary_tf} | {tf_decision.reason}")
            if tf_decision.context_filter:
                print(f"  Context: {tf_decision.context_filter}")
        else:
            # Без селектора - используем primary_tf из конфига
            self.current_primary_tf = self.config.primary_tf
        
        # ШАГ 3: Получаем AI сигнал для PRIMARY_TF
        ai_signal = multitf_signals.get(self.current_primary_tf)
        if not ai_signal:
            return
        
        self.current_ai_signal = ai_signal
        self.current_regime = ai_signal.get("regime", "unknown")
        
        # ШАГ 4: Генерация торгового сигнала на PRIMARY_TF
        primary_tf_data = self.multitf_data.get(self.current_primary_tf, historical_data)
        trading_signal = self._generate_trading_signal(primary_tf_data, ai_signal)
        
        if trading_signal:
            self._process_trading_signal(trading_signal, timestamp)
    
    def on_tick(self, tick: Tick):
        """
        Обработка тика (real-time или backtesting)
        
        Args:
            tick: Тиковые данные
        """
        # Проверка pending orders
        self._check_pending_orders(tick)
        
        # Обновление позиций (SL/TP/Trailing)
        self._update_positions(tick)
    
    def on_m1_candle(self, m1_candle: IntrabarCandle):
        """
        Обработка M1 свечи для интрабарной симуляции
        
        Args:
            m1_candle: M1 свеча
        """
        # Генерация тиков из M1
        ticks = m1_candle.to_ticks(num_ticks=10)
        
        for tick in ticks:
            self.on_tick(tick)
    
    def _request_ai_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Запрос сигнала от AI Core (один таймфрейм) - deprecated, использовать _request_multitf_signals"""
        # Формирование последних N свечей
        candles = []
        for idx, row in data.tail(100).iterrows():
            candles.append({
                "timestamp": str(row.name),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0))
            })
        
        return self.ai_client.predict(
            symbol=self.config.symbol,
            timeframe=self.config.base_timeframe,
            candles=candles
        )
    
    def _request_multitf_signals(self) -> Optional[Dict[str, Dict]]:
        """
        Запрос AI сигналов по всем таймфреймам.
        
        Returns:
            {tf: {regime, direction, direction_confidence, sentiment, ...}}
        """
        timeframes_data = {}
        
        # Формируем данные для каждого TF
        for tf in ["M5", "M15", "H1", "H4"]:
            if tf in self.multitf_data:
                data = self.multitf_data[tf]
            elif self.data is not None:
                # Fallback: используем текущие данные
                data = self.data
            else:
                continue
            
            candles = []
            for idx, row in data.tail(100).iterrows():
                candles.append({
                    "timestamp": str(row.name),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0))
                })
            
            timeframes_data[tf] = candles
        
        # Запрашиваем сигналы по всем TF
        return self.ai_client.predict_multitimeframe(
            symbol=self.config.symbol,
            timeframes_data=timeframes_data
        )
    
    def _select_primary_timeframe(self, multitf_signals: Dict[str, Dict]) -> TimeframeDecision:
        """
        Выбор PRIMARY_TF через TimeframeSelector.
        
        Args:
            multitf_signals: Сигналы от AI по всем TF
        
        Returns:
            TimeframeDecision
        """
        # Преобразуем AI сигналы в TimeframeData
        tf_data = {}
        
        for tf_str, signal in multitf_signals.items():
            try:
                tf = Timeframe(tf_str)
                regime_str = signal.get("regime", "unknown")
                
                # Преобразуем regime в Regime enum
                regime = Regime.UNKNOWN
                if regime_str == "trend_up":
                    regime = Regime.TREND_UP
                elif regime_str == "trend_down":
                    regime = Regime.TREND_DOWN
                elif regime_str == "range":
                    regime = Regime.RANGE
                elif regime_str == "volatile":
                    regime = Regime.VOLATILE
                
                tf_data[tf] = TimeframeData(
                    timeframe=tf,
                    regime=regime,
                    direction=signal.get("direction", "flat"),
                    direction_confidence=signal.get("direction_confidence", 0.0),
                    volatility_score=signal.get("volatility_score"),
                )
            except ValueError:
                continue
        
        # Вызываем селектор
        return self.tf_selector.select_timeframe(tf_data)
    
    def _generate_trading_signal(self, data: pd.DataFrame, ai_signal: Dict) -> Optional[Dict]:
        """
        Генерация торгового сигнала на основе AI и режима
        
        Returns:
            Signal dict или None
        """
        regime = ai_signal.get("regime", "unknown")
        
        # Выбор стратегии по режиму
        strategy = self.strategies.get(regime)
        if not strategy:
            return None
        
        # Проверка возможности открытия позиции
        can_trade, reason = self.risk_manager.can_open_position(datetime.now())
        if not can_trade:
            print(f"Cannot trade: {reason}")
            return None
        
        # Генерация сигнала
        signal = strategy.generate_signal(data, ai_signal)
        
        return signal
    
    def _process_trading_signal(self, signal: Dict, timestamp: datetime):
        """Обработка торгового сигнала"""
        signal_type = signal["type"]
        
        # Расчёт размера позиции
        risk_reduction = signal.get("risk_reduction", 1.0)
        volume = self.risk_manager.calculate_position_size(
            entry_price=signal["price"],
            sl_price=signal["sl"],
            direction="long" if "buy" in signal_type else "short",
            risk_reduction=risk_reduction
        )
        
        # Создание ордера
        order = {
            "type": signal_type,
            "price": signal["price"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "volume": volume,
            "reason": signal["reason"],
            "regime": signal["regime"],
            "confidence": signal["confidence"],
            "timestamp": timestamp
        }
        
        # Добавление в pending orders
        order_id = f"O{len(self.pending_orders) + 1:05d}"
        self.pending_orders[order_id] = order
        
        print(f"[{timestamp}] New order: {signal_type} @ {signal['price']:.2f}, "
              f"SL={signal['sl']:.2f}, TP={signal['tp']:.2f}, Vol={volume:.2f}")
        print(f"  Reason: {signal['reason']}")
    
    def _check_pending_orders(self, tick: Tick):
        """Проверка и исполнение pending orders"""
        executed_orders = []
        
        for order_id, order in self.pending_orders.items():
            position = self.intrabar_engine.simulate_order_execution(order, tick)
            
            if position:
                # Ордер исполнен
                trade = Trade(
                    id="",  # Будет присвоен в risk_manager
                    symbol=self.config.symbol,
                    direction=position["type"],
                    entry_price=position["entry"],
                    entry_time=tick.timestamp,
                    volume=position["volume"],
                    sl=position["sl"],
                    tp=position["tp"],
                    regime=order["regime"],
                    reason=order["reason"]
                )
                
                trade_id = self.risk_manager.open_position(trade)
                executed_orders.append(order_id)
                
                print(f"[{tick.timestamp}] Position opened: {trade_id}, "
                      f"{position['type']} @ {position['entry']:.2f}")
        
        # Удаление исполненных ордеров
        for order_id in executed_orders:
            del self.pending_orders[order_id]
    
    def _update_positions(self, tick: Tick):
        """Обновление открытых позиций (SL/TP/Trailing)"""
        closed_positions = []
        
        for trade_id, position in self.risk_manager.open_positions.items():
            # Проверка Stop Loss
            if self.intrabar_engine.check_stop_loss(position.__dict__, tick):
                self.risk_manager.close_position(trade_id, position.sl, tick.timestamp)
                closed_positions.append(trade_id)
                print(f"[{tick.timestamp}] SL hit: {trade_id} @ {position.sl:.2f}")
                continue
            
            # Проверка Take Profit
            if self.intrabar_engine.check_take_profit(position.__dict__, tick):
                self.risk_manager.close_position(trade_id, position.tp, tick.timestamp)
                closed_positions.append(trade_id)
                print(f"[{tick.timestamp}] TP hit: {trade_id} @ {position.tp:.2f}")
                continue
            
            # Trailing Stop (только для trend режима)
            if position.regime in ["trend_up", "trend_down"]:
                strategy = self.strategies.get(position.regime)
                
                if isinstance(strategy, TrendStrategy):
                    current_price = tick.bid if position.direction == "long" else tick.ask
                    
                    if strategy.should_trail_stop(position.__dict__, current_price):
                        # Получаем ATR
                        atr = self.data.iloc[-1].get("atr", 100.0) if self.data is not None else 100.0
                        trailing_distance = strategy.calculate_trailing_distance(atr)
                        
                        new_sl = self.intrabar_engine.update_trailing_stop(
                            position.__dict__, tick, trailing_distance
                        )
                        
                        if new_sl:
                            position.sl = new_sl
                            print(f"[{tick.timestamp}] Trailing SL updated: {trade_id} -> {new_sl:.2f}")
        
        # Отправка feedback в AI
        for trade_id in closed_positions:
            self._send_trade_feedback(trade_id)
    
    def _send_trade_feedback(self, trade_id: str):
        """Отправка feedback в AI после закрытия сделки"""
        trade = next((t for t in self.risk_manager.trade_history if t.id == trade_id), None)
        
        if trade and trade.pnl is not None:
            feedback = {
                "symbol": self.config.symbol,
                "regime": trade.regime,
                "direction": trade.direction,
                "sentiment": self.current_ai_signal.get("sentiment", 0) if self.current_ai_signal else 0,
                "result_pnl": trade.pnl,
                "good_trade": trade.pnl > 0
            }
            
            self.ai_client.send_feedback(feedback)
    
    def get_statistics(self) -> Dict:
        """Получение статистики"""
        stats = {
            **self.risk_manager.get_statistics(),
            "regime_stats": self.risk_manager.get_regime_statistics(),
            "current_regime": self.current_regime,
            "current_primary_tf": self.current_primary_tf,
            "open_positions": len(self.risk_manager.open_positions),
            "pending_orders": len(self.pending_orders)
        }
        
        # Добавляем информацию о текущем решении TimeframeSelector
        if self.current_tf_decision:
            stats["tf_decision"] = {
                "primary_tf": self.current_tf_decision.primary_tf.value,
                "reason": self.current_tf_decision.reason,
                "should_trade": self.current_tf_decision.should_trade,
                "context_filter": self.current_tf_decision.context_filter
            }
        
        return stats

"""
Модуль выбора таймфрейма (Timeframe Selector) для Golden Breeze Hybrid Strategy v1.0.

Отвечает за динамический выбор PRIMARY_TF на основе:
- regime и confidence по разным таймфреймам
- контекста старших таймфреймов (H1/H4)
- волатильности

Автор: Golden Breeze Team
Версия: 1.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Timeframe(str, Enum):
    """Поддерживаемые таймфреймы."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"


class Regime(str, Enum):
    """Режимы рынка."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class TimeframeData:
    """Данные по конкретному таймфрейму."""
    timeframe: Timeframe
    regime: Regime
    direction: str  # 'long', 'short', 'flat'
    direction_confidence: float  # 0..1
    volatility_score: Optional[float] = None  # опционально
    atr_value: Optional[float] = None  # опционально


@dataclass
class TimeframeDecision:
    """Решение селектора таймфреймов."""
    primary_tf: Timeframe  # рабочий таймфрейм для сигналов
    context_tf_high: Timeframe  # старший контекстный TF (H1/H4)
    exec_tf_low: Timeframe  # младший TF для исполнения (M1)
    reason: str  # причина выбора PRIMARY_TF
    should_trade: bool  # можно ли вообще торговать сейчас
    context_filter: Optional[str] = None  # дополнительные фильтры от старших TF


class TimeframeSelector:
    """
    Модуль выбора таймфрейма.
    
    Принимает данные по всем таймфреймам (M5, M15, H1, H4)
    и выбирает оптимальный PRIMARY_TF для текущей рыночной ситуации.
    """
    
    def __init__(
        self,
        default_primary_tf: Timeframe = Timeframe.M5,
        min_confidence_threshold: float = 0.65,
        high_confidence_threshold: float = 0.8,
        context_tf_high: Timeframe = Timeframe.H1,
        exec_tf_low: Timeframe = Timeframe.M1
    ):
        """
        Args:
            default_primary_tf: таймфрейм по умолчанию
            min_confidence_threshold: минимальная уверенность для работы на TF
            high_confidence_threshold: высокая уверенность (усиленный приоритет)
            context_tf_high: старший контекстный таймфрейм
            exec_tf_low: младший таймфрейм для исполнения
        """
        self.default_primary_tf = default_primary_tf
        self.min_confidence = min_confidence_threshold
        self.high_confidence = high_confidence_threshold
        self.context_tf_high = context_tf_high
        self.exec_tf_low = exec_tf_low
        
        # Текущий выбранный PRIMARY_TF (может меняться)
        self.current_primary_tf = default_primary_tf
        
        # История решений (для анализа/отладки)
        self.decision_history: List[TimeframeDecision] = []
    
    def select_timeframe(
        self,
        tf_data: Dict[Timeframe, TimeframeData]
    ) -> TimeframeDecision:
        """
        Главный метод: выбирает PRIMARY_TF на основе данных по всем таймфреймам.
        
        Args:
            tf_data: словарь {Timeframe: TimeframeData} с данными по M5, M15, H1, H4
        
        Returns:
            TimeframeDecision с выбранным PRIMARY_TF и контекстом
        """
        # Проверяем наличие необходимых данных
        if Timeframe.M5 not in tf_data or Timeframe.M15 not in tf_data:
            return self._create_no_trade_decision("Нет данных по M5 или M15")
        
        # Получаем данные по ключевым таймфреймам
        m5_data = tf_data.get(Timeframe.M5)
        m15_data = tf_data.get(Timeframe.M15)
        h1_data = tf_data.get(Timeframe.H1)
        h4_data = tf_data.get(Timeframe.H4)
        
        # Шаг 1: Проверяем контекст старших таймфреймов
        context_filter = self._analyze_high_timeframes(h1_data, h4_data)
        
        # Если старшие TF в хаосе, снижаем активность
        if context_filter and "высокая волатильность" in context_filter:
            # Можем торговать только при очень высокой уверенности
            required_confidence = self.high_confidence
        else:
            required_confidence = self.min_confidence
        
        # Шаг 2: Выбираем PRIMARY_TF по правилам
        selected_tf, reason = self._apply_selection_rules(
            m5_data, m15_data, h1_data, h4_data, required_confidence
        )
        
        # Шаг 3: Определяем, можно ли торговать
        should_trade = self._should_allow_trading(
            selected_tf, tf_data.get(selected_tf), context_filter
        )
        
        # Создаём решение
        decision = TimeframeDecision(
            primary_tf=selected_tf,
            context_tf_high=self.context_tf_high,
            exec_tf_low=self.exec_tf_low,
            reason=reason,
            should_trade=should_trade,
            context_filter=context_filter
        )
        
        # Сохраняем в историю
        self.decision_history.append(decision)
        self.current_primary_tf = selected_tf
        
        return decision
    
    def _apply_selection_rules(
        self,
        m5_data: TimeframeData,
        m15_data: TimeframeData,
        h1_data: Optional[TimeframeData],
        h4_data: Optional[TimeframeData],
        required_confidence: float
    ) -> tuple[Timeframe, str]:
        """
        Применяет правила выбора PRIMARY_TF (v1.0, простые правила).
        
        Returns:
            (selected_timeframe, reason)
        """
        # Правило 1: Проверяем M5
        if (m5_data.direction_confidence >= required_confidence and
            m5_data.regime in [Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE]):
            
            # Дополнительная проверка: согласуется ли с H1?
            if h1_data and self._check_tf_alignment(m5_data, h1_data):
                return (
                    Timeframe.M5,
                    f"M5 confidence={m5_data.direction_confidence:.2f}, "
                    f"regime={m5_data.regime.value}, aligned with H1"
                )
            else:
                return (
                    Timeframe.M5,
                    f"M5 confidence={m5_data.direction_confidence:.2f}, "
                    f"regime={m5_data.regime.value}"
                )
        
        # Правило 2: Если M5 не подходит, проверяем M15
        if (m15_data.direction_confidence >= required_confidence and
            m15_data.regime in [Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE]):
            
            return (
                Timeframe.M15,
                f"M5 недостаточно уверен, переход на M15 "
                f"(confidence={m15_data.direction_confidence:.2f})"
            )
        
        # Правило 3: Если оба младших TF плохи, но H1 уверен
        if (h1_data and 
            h1_data.direction_confidence >= self.high_confidence and
            h1_data.regime in [Regime.TREND_UP, Regime.TREND_DOWN]):
            
            return (
                Timeframe.H1,
                f"M5/M15 в неопределённости, H1 показывает чёткий тренд "
                f"(confidence={h1_data.direction_confidence:.2f})"
            )
        
        # Правило 4: По умолчанию остаёмся на текущем PRIMARY_TF
        return (
            self.current_primary_tf,
            f"Сохраняем текущий TF={self.current_primary_tf.value}, "
            f"условия неопределённые"
        )
    
    def _analyze_high_timeframes(
        self,
        h1_data: Optional[TimeframeData],
        h4_data: Optional[TimeframeData]
    ) -> Optional[str]:
        """
        Анализирует старшие таймфреймы для определения контекстных фильтров.
        
        Returns:
            Строка с описанием фильтров или None
        """
        filters = []
        
        # Проверяем H4
        if h4_data:
            if h4_data.regime == Regime.VOLATILE:
                filters.append("H4: высокая волатильность")
            elif h4_data.regime in [Regime.TREND_UP, Regime.TREND_DOWN]:
                filters.append(f"H4: тренд {h4_data.direction}")
        
        # Проверяем H1
        if h1_data:
            if h1_data.regime == Regime.VOLATILE:
                filters.append("H1: волатильность")
            elif h1_data.regime in [Regime.TREND_UP, Regime.TREND_DOWN]:
                filters.append(f"H1: тренд {h1_data.direction}")
        
        return " | ".join(filters) if filters else None
    
    def _check_tf_alignment(
        self,
        lower_tf: TimeframeData,
        higher_tf: TimeframeData
    ) -> bool:
        """
        Проверяет согласованность направлений двух таймфреймов.
        
        Returns:
            True если направления совпадают
        """
        # Если оба показывают один тренд
        if (lower_tf.regime in [Regime.TREND_UP, Regime.TREND_DOWN] and
            higher_tf.regime in [Regime.TREND_UP, Regime.TREND_DOWN]):
            return lower_tf.direction == higher_tf.direction
        
        # Если младший TF в рэндже, а старший в тренде - это OK
        if lower_tf.regime == Regime.RANGE:
            return True
        
        return False
    
    def _should_allow_trading(
        self,
        primary_tf: Timeframe,
        primary_data: Optional[TimeframeData],
        context_filter: Optional[str]
    ) -> bool:
        """
        Определяет, можно ли открывать новые сделки в текущих условиях.
        
        Returns:
            True если торговля разрешена
        """
        if not primary_data:
            return False
        
        # Не торгуем при чистом хаосе
        if primary_data.regime == Regime.VOLATILE:
            # Разве что уверенность экстремально высокая
            if primary_data.direction_confidence < self.high_confidence:
                return False
        
        # Если старшие TF показывают высокую волатильность
        if context_filter and "высокая волатильность" in context_filter:
            # Требуем повышенную уверенность
            if primary_data.direction_confidence < self.high_confidence:
                return False
        
        # Базовая проверка: достаточная уверенность
        if primary_data.direction_confidence < self.min_confidence:
            return False
        
        return True
    
    def _create_no_trade_decision(self, reason: str) -> TimeframeDecision:
        """Создаёт решение с запретом торговли."""
        return TimeframeDecision(
            primary_tf=self.current_primary_tf,
            context_tf_high=self.context_tf_high,
            exec_tf_low=self.exec_tf_low,
            reason=reason,
            should_trade=False,
            context_filter="NO TRADE"
        )
    
    def get_current_primary_tf(self) -> Timeframe:
        """Возвращает текущий PRIMARY_TF."""
        return self.current_primary_tf
    
    def get_decision_history(self) -> List[TimeframeDecision]:
        """Возвращает историю решений селектора."""
        return self.decision_history
    
    def reset_history(self):
        """Очищает историю решений."""
        self.decision_history.clear()

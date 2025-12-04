"""
Golden Breeze v4 - Strategy Signals Generator

Генерирует сигналы от нескольких классических торговых стратегий.
Модель учится комбинировать эти сигналы, а не изобретать колесо.

Включённые стратегии:
1. EMA Crossover (9/21, 20/50)
2. RSI Oversold/Overbought
3. MACD Signal
4. Bollinger Bands Breakout
5. Support/Resistance Levels
6. Candlestick Patterns
7. Volume Confirmation
8. ATR-based Volatility Filter

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategySignal:
    """Сигнал от стратегии."""
    name: str
    signal: int  # -1 = SELL, 0 = NEUTRAL, 1 = BUY
    strength: float  # 0.0 - 1.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.signal, self.strength], dtype=np.float32)


class StrategySignalsGenerator:
    """
    Генератор сигналов от классических стратегий.
    
    Каждая стратегия возвращает:
    - signal: -1 (SELL), 0 (NEUTRAL), 1 (BUY)
    - strength: 0.0 - 1.0 (уверенность сигнала)
    
    Пример использования:
        >>> gen = StrategySignalsGenerator()
        >>> signals = gen.generate_all_signals(df_m5, df_h1)
        >>> # signals - DataFrame с колонками для каждой стратегии
    """
    
    def __init__(
        self,
        ema_fast_periods: List[int] = [9, 20],
        ema_slow_periods: List[int] = [21, 50],
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
    ):
        self.ema_fast_periods = ema_fast_periods
        self.ema_slow_periods = ema_slow_periods
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
    
    # =========================================================================
    # 1. EMA CROSSOVER STRATEGY
    # =========================================================================
    
    def ema_crossover_signal(
        self, 
        df: pd.DataFrame,
        fast_period: int,
        slow_period: int,
    ) -> pd.DataFrame:
        """
        EMA Crossover Strategy.
        
        BUY: Fast EMA crosses above Slow EMA
        SELL: Fast EMA crosses below Slow EMA
        
        Strength = расстояние между EMA (нормализованное)
        """
        close = df['close']
        
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        # Сигнал: положение Fast относительно Slow
        diff = ema_fast - ema_slow
        diff_pct = diff / close * 100  # В процентах от цены
        
        # Crossover detection
        cross_up = (diff > 0) & (diff.shift(1) <= 0)
        cross_down = (diff < 0) & (diff.shift(1) >= 0)
        
        signal = pd.Series(0, index=df.index)
        signal[diff > 0] = 1
        signal[diff < 0] = -1
        
        # Strength = абсолютное расстояние, нормализованное
        strength = np.abs(diff_pct).clip(0, 2) / 2  # max 2% = strength 1.0
        
        # Boost strength on crossover
        strength[cross_up | cross_down] = 1.0
        
        return pd.DataFrame({
            f'ema_{fast_period}_{slow_period}_signal': signal,
            f'ema_{fast_period}_{slow_period}_strength': strength,
        }, index=df.index)
    
    # =========================================================================
    # 2. RSI STRATEGY
    # =========================================================================
    
    def rsi_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI Oversold/Overbought Strategy.
        
        BUY: RSI < 30 (oversold)
        SELL: RSI > 70 (overbought)
        
        Strength = насколько далеко от нейтральной зоны
        """
        close = df['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        signal = pd.Series(0, index=df.index)
        signal[rsi < self.rsi_oversold] = 1   # Oversold = BUY
        signal[rsi > self.rsi_overbought] = -1  # Overbought = SELL
        
        # Strength: насколько экстремальный RSI
        strength = pd.Series(0.0, index=df.index)
        
        # Для oversold: чем ниже RSI, тем сильнее
        oversold_mask = rsi < self.rsi_oversold
        strength[oversold_mask] = (self.rsi_oversold - rsi[oversold_mask]) / self.rsi_oversold
        
        # Для overbought: чем выше RSI, тем сильнее
        overbought_mask = rsi > self.rsi_overbought
        strength[overbought_mask] = (rsi[overbought_mask] - self.rsi_overbought) / (100 - self.rsi_overbought)
        
        strength = strength.clip(0, 1)
        
        return pd.DataFrame({
            'rsi_signal': signal,
            'rsi_strength': strength,
            'rsi_value': rsi / 100,  # Нормализованное значение RSI
        }, index=df.index)
    
    # =========================================================================
    # 3. MACD STRATEGY
    # =========================================================================
    
    def macd_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD Strategy.
        
        BUY: MACD crosses above Signal line
        SELL: MACD crosses below Signal line
        
        Strength = размер гистограммы
        """
        close = df['close']
        
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Сигнал на основе гистограммы
        signal = pd.Series(0, index=df.index)
        signal[histogram > 0] = 1
        signal[histogram < 0] = -1
        
        # Crossover boost
        cross_up = (histogram > 0) & (histogram.shift(1) <= 0)
        cross_down = (histogram < 0) & (histogram.shift(1) >= 0)
        
        # Strength = размер гистограммы относительно цены
        hist_pct = np.abs(histogram) / close * 100
        strength = hist_pct.clip(0, 1)  # max 1% = strength 1.0
        strength[cross_up | cross_down] = 1.0
        
        return pd.DataFrame({
            'macd_signal': signal,
            'macd_strength': strength,
            'macd_histogram': histogram / close,  # Нормализованная гистограмма
        }, index=df.index)
    
    # =========================================================================
    # 4. BOLLINGER BANDS STRATEGY
    # =========================================================================
    
    def bollinger_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands Strategy.
        
        BUY: Цена касается нижней полосы (потенциальный отскок)
        SELL: Цена касается верхней полосы (потенциальный отскок)
        
        Также отслеживаем сжатие полос (squeeze) как признак скорого движения.
        """
        close = df['close']
        
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std
        
        # Позиция цены относительно полос (0 = нижняя, 1 = верхняя)
        bb_position = (close - lower_band) / (upper_band - lower_band + 1e-10)
        bb_position = bb_position.clip(0, 1)
        
        # Сигналы mean reversion
        signal = pd.Series(0, index=df.index)
        signal[bb_position < 0.1] = 1   # У нижней полосы = BUY
        signal[bb_position > 0.9] = -1  # У верхней полосы = SELL
        
        # Strength = насколько близко к полосе
        strength = pd.Series(0.0, index=df.index)
        strength[bb_position < 0.1] = (0.1 - bb_position[bb_position < 0.1]) / 0.1
        strength[bb_position > 0.9] = (bb_position[bb_position > 0.9] - 0.9) / 0.1
        strength = strength.clip(0, 1)
        
        # Ширина полос (для squeeze detection)
        bb_width = (upper_band - lower_band) / sma
        bb_squeeze = 1 - (bb_width / bb_width.rolling(50).max()).clip(0, 1)
        
        return pd.DataFrame({
            'bb_signal': signal,
            'bb_strength': strength,
            'bb_position': bb_position,
            'bb_squeeze': bb_squeeze.fillna(0),
        }, index=df.index)
    
    # =========================================================================
    # 5. SUPPORT/RESISTANCE LEVELS
    # =========================================================================
    
    def support_resistance_signal(
        self, 
        df: pd.DataFrame,
        lookback: int = 50,
    ) -> pd.DataFrame:
        """
        Support/Resistance Strategy.
        
        Определяет локальные уровни поддержки и сопротивления.
        BUY: Цена у поддержки
        SELL: Цена у сопротивления
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Локальные максимумы и минимумы
        roll_high = high.rolling(lookback).max()
        roll_low = low.rolling(lookback).min()
        
        # Расстояние до уровней
        dist_to_resistance = (roll_high - close) / close
        dist_to_support = (close - roll_low) / close
        
        # Сигналы
        signal = pd.Series(0, index=df.index)
        
        # У поддержки (в пределах 0.5% от минимума)
        near_support = dist_to_support < 0.005
        signal[near_support] = 1
        
        # У сопротивления (в пределах 0.5% от максимума)
        near_resistance = dist_to_resistance < 0.005
        signal[near_resistance] = -1
        
        # Strength = насколько близко
        strength = pd.Series(0.0, index=df.index)
        strength[near_support] = 1 - dist_to_support[near_support] / 0.005
        strength[near_resistance] = 1 - dist_to_resistance[near_resistance] / 0.005
        strength = strength.clip(0, 1)
        
        return pd.DataFrame({
            'sr_signal': signal,
            'sr_strength': strength,
            'dist_to_support': dist_to_support.clip(0, 0.1),
            'dist_to_resistance': dist_to_resistance.clip(0, 0.1),
        }, index=df.index)
    
    # =========================================================================
    # 6. CANDLESTICK PATTERNS
    # =========================================================================
    
    def candlestick_patterns_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Candlestick Patterns Strategy.
        
        Обнаруживает классические свечные паттерны:
        - Hammer / Hanging Man
        - Engulfing (бычий/медвежий)
        - Doji
        - Pin Bar
        """
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        body = close - open_
        body_abs = np.abs(body)
        upper_shadow = high - np.maximum(open_, close)
        lower_shadow = np.minimum(open_, close) - low
        total_range = high - low
        
        # Средний размер свечи для нормализации
        avg_body = body_abs.rolling(20).mean()
        
        signal = pd.Series(0.0, index=df.index)
        strength = pd.Series(0.0, index=df.index)
        pattern_type = pd.Series(0, index=df.index)
        
        # 1. Hammer (бычий) - маленькое тело наверху, длинная нижняя тень
        hammer = (
            (lower_shadow > 2 * body_abs) & 
            (upper_shadow < body_abs * 0.5) &
            (body > 0)  # Закрытие выше открытия
        )
        signal[hammer] = 1
        strength[hammer] = 0.8
        pattern_type[hammer] = 1
        
        # 2. Hanging Man (медвежий) - как hammer, но на вершине тренда
        hanging_man = (
            (lower_shadow > 2 * body_abs) & 
            (upper_shadow < body_abs * 0.5) &
            (body < 0)
        )
        signal[hanging_man] = -1
        strength[hanging_man] = 0.8
        pattern_type[hanging_man] = 2
        
        # 3. Bullish Engulfing
        prev_body = body.shift(1)
        bullish_engulf = (
            (body > 0) & 
            (prev_body < 0) &
            (body_abs > np.abs(prev_body)) &
            (open_ < close.shift(1)) &
            (close > open_.shift(1))
        )
        signal[bullish_engulf] = 1
        strength[bullish_engulf] = 0.9
        pattern_type[bullish_engulf] = 3
        
        # 4. Bearish Engulfing
        bearish_engulf = (
            (body < 0) & 
            (prev_body > 0) &
            (body_abs > np.abs(prev_body)) &
            (open_ > close.shift(1)) &
            (close < open_.shift(1))
        )
        signal[bearish_engulf] = -1
        strength[bearish_engulf] = 0.9
        pattern_type[bearish_engulf] = 4
        
        # 5. Doji (неопределённость)
        doji = body_abs < (total_range * 0.1)
        strength[doji] = 0.3  # Слабый сигнал неопределённости
        pattern_type[doji] = 5
        
        # 6. Pin Bar (rejection)
        pin_bar_bull = (
            (lower_shadow > 2.5 * body_abs) & 
            (lower_shadow > upper_shadow * 2)
        )
        signal[pin_bar_bull] = 1
        strength[pin_bar_bull] = 0.85
        pattern_type[pin_bar_bull] = 6
        
        pin_bar_bear = (
            (upper_shadow > 2.5 * body_abs) & 
            (upper_shadow > lower_shadow * 2)
        )
        signal[pin_bar_bear] = -1
        strength[pin_bar_bear] = 0.85
        pattern_type[pin_bar_bear] = 7
        
        return pd.DataFrame({
            'candle_signal': signal,
            'candle_strength': strength,
            'candle_pattern': pattern_type,
            'candle_body_ratio': (body_abs / (total_range + 1e-10)).clip(0, 1),
        }, index=df.index)
    
    # =========================================================================
    # 7. VOLUME CONFIRMATION
    # =========================================================================
    
    def volume_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume Confirmation Strategy.
        
        Высокий объём подтверждает движение.
        Низкий объём на откате - признак продолжения тренда.
        """
        close = df['close']
        
        # Определяем колонку объёма
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
        elif 'volume' in df.columns:
            volume = df['volume']
        else:
            # Нет объёма - возвращаем нули
            return pd.DataFrame({
                'volume_signal': 0,
                'volume_strength': 0,
                'volume_ratio': 0,
            }, index=df.index)
        
        # Средний объём
        avg_volume = volume.rolling(20).mean()
        volume_ratio = volume / (avg_volume + 1e-10)
        
        # Изменение цены
        price_change = close.diff()
        
        # Высокий объём с движением = подтверждение
        high_volume = volume_ratio > 1.5
        
        signal = pd.Series(0, index=df.index)
        signal[(high_volume) & (price_change > 0)] = 1   # Volume confirms up
        signal[(high_volume) & (price_change < 0)] = -1  # Volume confirms down
        
        # Strength = насколько выше среднего
        strength = ((volume_ratio - 1) / 2).clip(0, 1)
        strength[~high_volume] = 0
        
        return pd.DataFrame({
            'volume_signal': signal,
            'volume_strength': strength,
            'volume_ratio': volume_ratio.clip(0, 5) / 5,  # Нормализуем
        }, index=df.index)
    
    # =========================================================================
    # 8. ATR VOLATILITY FILTER
    # =========================================================================
    
    def atr_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR Volatility Filter.
        
        Не даёт торговые сигналы, но показывает:
        - Текущую волатильность
        - Фазу рынка (низкая/высокая волатильность)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # ATR как процент от цены
        atr_pct = atr / close
        
        # Сравнение с историческим ATR
        atr_avg = atr.rolling(50).mean()
        atr_ratio = atr / (atr_avg + 1e-10)
        
        # Высокая волатильность = возможность для трейда
        # Но также и риск
        volatility_state = pd.Series(0, index=df.index)
        volatility_state[atr_ratio > 1.5] = 1   # High volatility
        volatility_state[atr_ratio < 0.7] = -1  # Low volatility (squeeze)
        
        return pd.DataFrame({
            'atr_pct': atr_pct.clip(0, 0.05) / 0.05,  # Нормализуем (max 5%)
            'atr_ratio': atr_ratio.clip(0, 3) / 3,
            'volatility_state': volatility_state,
        }, index=df.index)
    
    # =========================================================================
    # 9. TREND STRENGTH (ADX-like)
    # =========================================================================
    
    def trend_strength_signal(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Trend Strength Indicator (simplified ADX).
        
        Показывает силу тренда и его направление.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Когда оба положительные, берём больший
        mask = plus_dm > minus_dm
        plus_dm[~mask] = 0
        minus_dm[mask] = 0
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        # Сигнал на основе DI crossover
        signal = pd.Series(0, index=df.index)
        signal[plus_di > minus_di] = 1   # Bullish trend
        signal[plus_di < minus_di] = -1  # Bearish trend
        
        # Strength = ADX (нормализованный)
        strength = (adx / 50).clip(0, 1)  # ADX > 50 = очень сильный тренд
        
        return pd.DataFrame({
            'trend_signal': signal,
            'trend_strength': strength,
            'adx': adx / 100,
            'plus_di': plus_di / 100,
            'minus_di': minus_di / 100,
        }, index=df.index)
    
    # =========================================================================
    # 10. SUPERTREND
    # =========================================================================
    
    def supertrend_signal(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        SuperTrend Indicator.
        
        Trend-following indicator combining ATR with price.
        BUY: Price above SuperTrend line
        SELL: Price below SuperTrend line
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR
        tr = pd.concat([
            high - low,
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # SuperTrend calculation
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 = up, -1 = down
        
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
                
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        # Signal
        signal = direction.copy()
        
        # Strength = distance from supertrend (normalized)
        dist = np.abs(close - supertrend) / (atr + 1e-10)
        strength = (dist / 2).clip(0, 1)
        
        return pd.DataFrame({
            'supertrend_signal': signal,
            'supertrend_strength': strength,
            'supertrend_dist': (close - supertrend) / close,
        }, index=df.index)
    
    # =========================================================================
    # 11. ICHIMOKU CLOUD
    # =========================================================================
    
    def ichimoku_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ichimoku Cloud Indicator.
        
        BUY: Price above cloud, Tenkan > Kijun
        SELL: Price below cloud, Tenkan < Kijun
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (9 periods)
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (26 periods)
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (shifted 26 periods)
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (52 periods, shifted 26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Cloud top and bottom
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        
        # Signal
        signal = pd.Series(0, index=df.index)
        
        # Above cloud = bullish
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom
        
        # Tenkan/Kijun cross
        tk_bullish = tenkan > kijun
        
        signal[above_cloud & tk_bullish] = 1
        signal[below_cloud & ~tk_bullish] = -1
        
        # Strength
        cloud_thickness = np.abs(senkou_a - senkou_b) / close
        price_cloud_dist = np.where(
            close > cloud_top,
            (close - cloud_top) / close,
            np.where(close < cloud_bottom, (cloud_bottom - close) / close, 0)
        )
        
        strength = (np.abs(price_cloud_dist) * 10).clip(0, 1)
        
        return pd.DataFrame({
            'ichimoku_signal': signal,
            'ichimoku_strength': strength,
            'ichimoku_tk_diff': (tenkan - kijun) / close,
            'ichimoku_cloud_pos': np.where(close > cloud_top, 1, np.where(close < cloud_bottom, -1, 0)),
        }, index=df.index)
    
    # =========================================================================
    # 12. VWAP (Volume Weighted Average Price)
    # =========================================================================
    
    def vwap_signal(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Rolling VWAP indicator.
        
        BUY: Price above VWAP (discount)
        SELL: Price below VWAP (premium)
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Get volume column
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        volume = df[vol_col] if vol_col in df.columns else pd.Series(1, index=df.index)
        
        # Typical price
        typical = (high + low + close) / 3
        
        # Rolling VWAP
        cum_vol = volume.rolling(period).sum()
        cum_tp_vol = (typical * volume).rolling(period).sum()
        vwap = cum_tp_vol / (cum_vol + 1e-10)
        
        # Signal
        diff = close - vwap
        diff_pct = diff / close
        
        signal = pd.Series(0, index=df.index)
        signal[diff_pct > 0.001] = 1   # Above VWAP
        signal[diff_pct < -0.001] = -1  # Below VWAP
        
        strength = (np.abs(diff_pct) * 50).clip(0, 1)
        
        return pd.DataFrame({
            'vwap_signal': signal,
            'vwap_strength': strength,
            'vwap_dist': diff_pct,
        }, index=df.index)
    
    # =========================================================================
    # 13. KELTNER CHANNELS
    # =========================================================================
    
    def keltner_signal(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Keltner Channels.
        
        Similar to Bollinger but uses ATR instead of std.
        """
        close = df['close']
        high = df['high']
        low = df['low']
        
        # EMA middle
        middle = close.ewm(span=period).mean()
        
        # ATR
        tr = pd.concat([
            high - low,
            np.abs(high - close.shift(1)),
            np.abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        # Position within channel
        channel_width = upper - lower
        position = (close - lower) / (channel_width + 1e-10)
        
        signal = pd.Series(0, index=df.index)
        signal[close > upper] = 1   # Breakout up
        signal[close < lower] = -1  # Breakout down
        
        # Strength = distance from middle
        strength = np.abs(position - 0.5) * 2
        
        return pd.DataFrame({
            'keltner_signal': signal,
            'keltner_strength': strength.clip(0, 1),
            'keltner_position': position.clip(0, 1),
        }, index=df.index)
    
    # =========================================================================
    # 14. CCI (Commodity Channel Index)
    # =========================================================================
    
    def cci_signal(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        CCI - Commodity Channel Index.
        
        BUY: CCI < -100 (oversold)
        SELL: CCI > 100 (overbought)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        typical = (high + low + close) / 3
        sma = typical.rolling(period).mean()
        mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical - sma) / (0.015 * mad + 1e-10)
        
        signal = pd.Series(0, index=df.index)
        signal[cci < -100] = 1   # Oversold
        signal[cci > 100] = -1   # Overbought
        
        strength = (np.abs(cci) / 200).clip(0, 1)
        
        return pd.DataFrame({
            'cci_signal': signal,
            'cci_strength': strength,
            'cci_value': (cci / 200).clip(-1, 1),
        }, index=df.index)
    
    # =========================================================================
    # 15. WILLIAMS %R
    # =========================================================================
    
    def williams_r_signal(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R - Momentum indicator.
        
        BUY: %R < -80 (oversold)
        SELL: %R > -20 (overbought)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        highest = high.rolling(period).max()
        lowest = low.rolling(period).min()
        
        williams_r = -100 * (highest - close) / (highest - lowest + 1e-10)
        
        signal = pd.Series(0, index=df.index)
        signal[williams_r < -80] = 1   # Oversold
        signal[williams_r > -20] = -1  # Overbought
        
        # Strength = distance from middle (-50)
        strength = np.abs(williams_r + 50) / 50
        
        return pd.DataFrame({
            'williams_signal': signal,
            'williams_strength': strength.clip(0, 1),
            'williams_value': (williams_r + 50) / 50,  # Normalize to -1 to 1
        }, index=df.index)
    
    # =========================================================================
    # 16. PARABOLIC SAR
    # =========================================================================
    
    def parabolic_sar_signal(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        Parabolic SAR - Trend following indicator.
        
        BUY: Price above SAR
        SELL: Price below SAR
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        
        psar = np.zeros(n)
        trend = np.ones(n)  # 1 = up, -1 = down
        af = np.full(n, af_start)
        ep = np.zeros(n)
        
        # Initialize
        psar[0] = low[0]
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:
                psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
                psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        signal = pd.Series(trend, index=df.index)
        dist = np.abs(close - psar) / close
        strength = (dist * 50).clip(0, 1)
        
        return pd.DataFrame({
            'psar_signal': signal,
            'psar_strength': strength,
            'psar_dist': (close - psar) / close,
        }, index=df.index)
    
    # =========================================================================
    # 17. AWESOME OSCILLATOR
    # =========================================================================
    
    def awesome_oscillator_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Awesome Oscillator - Bill Williams indicator.
        
        AO = SMA5(median) - SMA34(median)
        """
        high = df['high']
        low = df['low']
        
        median = (high + low) / 2
        ao = median.rolling(5).mean() - median.rolling(34).mean()
        
        # Signal
        signal = pd.Series(0, index=df.index)
        signal[ao > 0] = 1
        signal[ao < 0] = -1
        
        # Zero cross
        cross_up = (ao > 0) & (ao.shift(1) <= 0)
        cross_down = (ao < 0) & (ao.shift(1) >= 0)
        
        # Strength
        ao_norm = ao / (df['close'] * 0.01)  # Normalize by 1% of price
        strength = np.abs(ao_norm).clip(0, 1)
        strength[cross_up | cross_down] = 1.0
        
        return pd.DataFrame({
            'ao_signal': signal,
            'ao_strength': strength,
            'ao_value': ao_norm.clip(-1, 1),
        }, index=df.index)
    
    # =========================================================================
    # 18. MFI (Money Flow Index)
    # =========================================================================
    
    def mfi_signal(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Money Flow Index - Volume-weighted RSI.
        
        BUY: MFI < 20 (oversold)
        SELL: MFI > 80 (overbought)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        volume = df[vol_col] if vol_col in df.columns else pd.Series(1, index=df.index)
        
        typical = (high + low + close) / 3
        raw_money_flow = typical * volume
        
        # Positive and negative flow
        flow_direction = typical.diff()
        positive_flow = raw_money_flow.where(flow_direction > 0, 0)
        negative_flow = raw_money_flow.where(flow_direction < 0, 0)
        
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
        
        signal = pd.Series(0, index=df.index)
        signal[mfi < 20] = 1   # Oversold
        signal[mfi > 80] = -1  # Overbought
        
        strength = np.abs(mfi - 50) / 50
        
        return pd.DataFrame({
            'mfi_signal': signal,
            'mfi_strength': strength.clip(0, 1),
            'mfi_value': mfi / 100,
        }, index=df.index)
    
    # =========================================================================
    # 19. HURST EXPONENT (Simplified)
    # =========================================================================
    
    def hurst_signal(self, df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
        """
        Simplified Hurst Exponent estimation.
        
        H < 0.5: Mean-reverting (range trading)
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        close = df['close']
        returns = close.pct_change()
        
        # Simplified R/S analysis
        def calc_hurst(series):
            if len(series) < 20:
                return 0.5
            
            mean = series.mean()
            std = series.std()
            if std < 1e-10:
                return 0.5
            
            cumdev = np.cumsum(series - mean)
            R = cumdev.max() - cumdev.min()
            S = std
            
            if S > 0 and R > 0:
                # Approximate Hurst
                return np.log(R / S) / np.log(len(series))
            return 0.5
        
        hurst = returns.rolling(period).apply(calc_hurst, raw=True)
        hurst = hurst.fillna(0.5)
        
        # Signal based on regime
        signal = pd.Series(0, index=df.index)
        signal[hurst > 0.6] = 1   # Trending - follow trend
        signal[hurst < 0.4] = -1  # Mean-reverting - fade moves
        
        # Strength = deviation from 0.5
        strength = np.abs(hurst - 0.5) * 2
        
        return pd.DataFrame({
            'hurst_signal': signal,
            'hurst_strength': strength.clip(0, 1),
            'hurst_value': hurst,
        }, index=df.index)
    
    # =========================================================================
    # MAIN GENERATOR
    # =========================================================================
    
    def generate_all_signals(
        self, 
        df: pd.DataFrame,
        include_raw_values: bool = True,
    ) -> pd.DataFrame:
        """
        Генерирует все сигналы для DataFrame.
        
        Args:
            df: OHLCV DataFrame
            include_raw_values: Включать ли сырые значения индикаторов
            
        Returns:
            DataFrame с сигналами от всех стратегий
        """
        signals = []
        
        # 1. EMA Crossovers
        for fast, slow in zip(self.ema_fast_periods, self.ema_slow_periods):
            signals.append(self.ema_crossover_signal(df, fast, slow))
        
        # 2. RSI
        signals.append(self.rsi_signal(df))
        
        # 3. MACD
        signals.append(self.macd_signal(df))
        
        # 4. Bollinger Bands
        signals.append(self.bollinger_signal(df))
        
        # 5. Support/Resistance
        signals.append(self.support_resistance_signal(df))
        
        # 6. Candlestick Patterns
        signals.append(self.candlestick_patterns_signal(df))
        
        # 7. Volume
        signals.append(self.volume_signal(df))
        
        # 8. ATR/Volatility
        signals.append(self.atr_signal(df))
        
        # 9. Trend Strength
        signals.append(self.trend_strength_signal(df))
        
        # 10. SuperTrend
        signals.append(self.supertrend_signal(df))
        
        # 11. Ichimoku Cloud
        signals.append(self.ichimoku_signal(df))
        
        # 12. VWAP
        signals.append(self.vwap_signal(df))
        
        # 13. Keltner Channels
        signals.append(self.keltner_signal(df))
        
        # 14. CCI
        signals.append(self.cci_signal(df))
        
        # 15. Williams %R
        signals.append(self.williams_r_signal(df))
        
        # 16. Parabolic SAR
        signals.append(self.parabolic_sar_signal(df))
        
        # 17. Awesome Oscillator
        signals.append(self.awesome_oscillator_signal(df))
        
        # 18. MFI
        signals.append(self.mfi_signal(df))
        
        # 19. Hurst Exponent
        signals.append(self.hurst_signal(df))
        
        # Combine all
        result = pd.concat(signals, axis=1)
        
        # Fill NaN
        result = result.fillna(0)
        
        return result
    
    def get_signal_names(self) -> List[str]:
        """Возвращает список имён всех сигналов."""
        return [
            # EMA
            'ema_9_21_signal', 'ema_9_21_strength',
            'ema_20_50_signal', 'ema_20_50_strength',
            # RSI
            'rsi_signal', 'rsi_strength', 'rsi_value',
            # MACD
            'macd_signal', 'macd_strength', 'macd_histogram',
            # Bollinger
            'bb_signal', 'bb_strength', 'bb_position', 'bb_squeeze',
            # S/R
            'sr_signal', 'sr_strength', 'dist_to_support', 'dist_to_resistance',
            # Candles
            'candle_signal', 'candle_strength', 'candle_pattern', 'candle_body_ratio',
            # Volume
            'volume_signal', 'volume_strength', 'volume_ratio',
            # ATR
            'atr_pct', 'atr_ratio', 'volatility_state',
            # Trend
            'trend_signal', 'trend_strength', 'adx', 'plus_di', 'minus_di',
            # SuperTrend
            'supertrend_signal', 'supertrend_strength', 'supertrend_dist',
            # Ichimoku
            'ichimoku_signal', 'ichimoku_strength', 'ichimoku_tk_diff', 'ichimoku_cloud_pos',
            # VWAP
            'vwap_signal', 'vwap_strength', 'vwap_dist',
            # Keltner
            'keltner_signal', 'keltner_strength', 'keltner_position',
            # CCI
            'cci_signal', 'cci_strength', 'cci_value',
            # Williams %R
            'williams_signal', 'williams_strength', 'williams_value',
            # Parabolic SAR
            'psar_signal', 'psar_strength', 'psar_dist',
            # Awesome Oscillator
            'ao_signal', 'ao_strength', 'ao_value',
            # MFI
            'mfi_signal', 'mfi_strength', 'mfi_value',
            # Hurst
            'hurst_signal', 'hurst_strength', 'hurst_value',
        ]
    
    def get_feature_dim(self) -> int:
        """Возвращает размерность выходного вектора признаков."""
        return len(self.get_signal_names())


def generate_strategy_features(
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Удобная функция для генерации стратегических сигналов.
    
    Args:
        df_m5: M5 OHLCV data
        df_h1: H1 OHLCV data
        
    Returns:
        (m5_signals, h1_signals) - DataFrames с сигналами
    """
    gen = StrategySignalsGenerator()
    
    m5_signals = gen.generate_all_signals(df_m5)
    h1_signals = gen.generate_all_signals(df_h1)
    
    return m5_signals, h1_signals


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Strategy Signals Generator - Test")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range(start='2025-01-01', periods=n, freq='5min')
    price = 2650 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': dates,
        'open': price + np.random.randn(n) * 0.5,
        'high': price + np.abs(np.random.randn(n)) * 2,
        'low': price - np.abs(np.random.randn(n)) * 2,
        'close': price + np.random.randn(n) * 0.5,
        'tick_volume': np.random.randint(100, 1000, n),
    })
    
    print(f"\nInput data: {len(df)} bars")
    
    # Generate signals
    gen = StrategySignalsGenerator()
    signals = gen.generate_all_signals(df)
    
    print(f"\nGenerated signals: {signals.shape}")
    print(f"Feature dimension: {gen.get_feature_dim()}")
    print(f"\nSignal columns:")
    for col in signals.columns:
        print(f"  - {col}")
    
    # Show sample
    print(f"\nSample signals (last 5 rows):")
    print(signals.tail())
    
    # Signal statistics
    print(f"\nSignal statistics:")
    signal_cols = [c for c in signals.columns if 'signal' in c]
    for col in signal_cols:
        buy = (signals[col] == 1).sum()
        sell = (signals[col] == -1).sum()
        neutral = (signals[col] == 0).sum()
        print(f"  {col}: BUY={buy}, SELL={sell}, NEUTRAL={neutral}")
    
    print("\n✅ Strategy Signals Generator test passed!")

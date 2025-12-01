# aimodule/inference/combine_signals.py

"""
Улучшенная ensemble логика принятия решения.

Правила:
- sentiment < -0.4 → SKIP всегда
- VOLATILE и confidence < 0.6 → SKIP
- RANGE и direction != FLAT, но confidence < 0.5 → HOLD
- TREND_UP и direction = LONG и sentiment > 0 → ENTER_LONG
- TREND_DOWN и direction = SHORT и sentiment < 0 → ENTER_SHORT
- Во всех спорных случаях → HOLD
"""

from typing import Tuple, List
from ..utils import (
    MarketRegime,
    Direction,
    Action,
)


def decide_action(
    regime: MarketRegime,
    direction: Direction,
    sentiment: float,
    confidence: float,
) -> Tuple[Action, List[str]]:
    """
    Комбинированная логика принятия решения с объяснениями.
    
    Args:
        regime: текущий режим рынка
        direction: предсказанное направление
        sentiment: оценка sentiment [-1, 1]
        confidence: уверенность модели [0, 1]
        
    Returns:
        (Action, reasons) - решение и список причин
    """
    reasons = []
    
    # Правило 1: Очень негативный sentiment → SKIP
    if sentiment < -0.4:
        reasons.append(f"Very negative sentiment ({sentiment:.2f} < -0.4)")
        reasons.append("Skipping trade due to poor market sentiment")
        return Action.SKIP, reasons
    
    # Правило 2: VOLATILE + низкая уверенность → SKIP
    if regime == MarketRegime.VOLATILE and confidence < 0.6:
        reasons.append(f"Volatile market regime detected")
        reasons.append(f"Low confidence ({confidence:.2%} < 60%) in volatile conditions")
        reasons.append("Skipping to avoid high-risk entry")
        return Action.SKIP, reasons
    
    # Правило 3: RANGE + несоответствие направления + низкая уверенность → HOLD
    if regime == MarketRegime.RANGE:
        if direction != Direction.FLAT and confidence < 0.5:
            reasons.append(f"Range-bound market (low volatility)")
            reasons.append(f"Confidence ({confidence:.2%}) below 50% threshold")
            reasons.append("Holding position, waiting for clearer signal")
            return Action.HOLD, reasons
        
        if abs(sentiment) < 0.1:
            reasons.append(f"Range-bound market with neutral sentiment")
            reasons.append("No clear directional bias, holding position")
            return Action.HOLD, reasons
    
    # Правило 4: TREND_UP + LONG + положительный sentiment → ENTER_LONG
    if regime == MarketRegime.TREND_UP:
        if direction == Direction.LONG and sentiment > 0:
            reasons.append(f"Strong uptrend detected (regime: {regime.value})")
            reasons.append(f"Direction model predicts LONG (confidence: {confidence:.2%})")
            reasons.append(f"Positive sentiment ({sentiment:.2f}) supports entry")
            return Action.ENTER_LONG, reasons
        
        if direction == Direction.LONG and sentiment >= -0.2:
            reasons.append(f"Uptrend with LONG signal (confidence: {confidence:.2%})")
            reasons.append(f"Sentiment ({sentiment:.2f}) is acceptable")
            return Action.ENTER_LONG, reasons
        
        if direction == Direction.SHORT:
            reasons.append(f"Uptrend conflicts with SHORT signal")
            reasons.append("Holding to avoid counter-trend trade")
            return Action.HOLD, reasons
    
    # Правило 5: TREND_DOWN + SHORT + негативный sentiment → ENTER_SHORT
    if regime == MarketRegime.TREND_DOWN:
        if direction == Direction.SHORT and sentiment < 0:
            reasons.append(f"Strong downtrend detected (regime: {regime.value})")
            reasons.append(f"Direction model predicts SHORT (confidence: {confidence:.2%})")
            reasons.append(f"Negative sentiment ({sentiment:.2f}) supports entry")
            return Action.ENTER_SHORT, reasons
        
        if direction == Direction.SHORT and sentiment <= 0.2:
            reasons.append(f"Downtrend with SHORT signal (confidence: {confidence:.2%})")
            reasons.append(f"Sentiment ({sentiment:.2f}) is acceptable")
            return Action.ENTER_SHORT, reasons
        
        if direction == Direction.LONG:
            reasons.append(f"Downtrend conflicts with LONG signal")
            reasons.append("Holding to avoid counter-trend trade")
            return Action.HOLD, reasons
    
    # Правило 6: Минимальная уверенность по режиму
    min_conf = 0.25
    if regime == MarketRegime.VOLATILE:
        min_conf = 0.35
    
    if confidence < min_conf:
        reasons.append(f"Confidence ({confidence:.2%}) below minimum threshold ({min_conf:.0%})")
        reasons.append(f"Regime: {regime.value}")
        reasons.append("Holding position due to low confidence")
        return Action.HOLD, reasons
    
    # Правило 7: Следование направлению при нормальных условиях
    if direction == Direction.LONG:
        reasons.append(f"Direction: LONG (confidence: {confidence:.2%})")
        reasons.append(f"Regime: {regime.value}, Sentiment: {sentiment:.2f}")
        reasons.append("Conditions acceptable for long entry")
        return Action.ENTER_LONG, reasons
    
    if direction == Direction.SHORT:
        reasons.append(f"Direction: SHORT (confidence: {confidence:.2%})")
        reasons.append(f"Regime: {regime.value}, Sentiment: {sentiment:.2f}")
        reasons.append("Conditions acceptable for short entry")
        return Action.ENTER_SHORT, reasons
    
    # Правило 8: Default → HOLD
    reasons.append(f"No clear signal: direction={direction.value}, regime={regime.value}")
    reasons.append(f"Confidence: {confidence:.2%}, Sentiment: {sentiment:.2f}")
    reasons.append("Holding position as default safe action")
    return Action.HOLD, reasons

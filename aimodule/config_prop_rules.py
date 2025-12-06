"""
Prop Firm Rules Configuration
Конфигурация правил для проп-компаний

TradersMastery via TradeLocker
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class AccountType(Enum):
    """Типы аккаунтов проп-компании"""
    EVALUATION = "evaluation"       # Оценочный этап
    ACCELERATED = "accelerated"     # Ускоренный
    MASTER = "master"               # Мастер (funded)


@dataclass
class TierRules:
    """Правила для конкретного тира"""
    tier_name: str
    min_balance: float
    max_balance: float
    
    # Лимиты просадки (в процентах)
    max_daily_loss_percent: float      # Макс. дневной убыток
    max_total_drawdown_percent: float  # Макс. общая просадка
    
    # Цели
    profit_target_percent: float       # Цель прибыли (0 для funded)
    
    # Торговые ограничения
    max_leverage: float                # Максимальное плечо
    max_positions: int                 # Макс. одновременных позиций
    max_lot_size: float                # Макс. размер лота
    
    # Дополнительные правила
    min_trading_days: int              # Минимум торговых дней
    weekend_holding: bool              # Можно ли держать через выходные
    news_trading: bool                 # Торговля на новостях


# ============================================================================
# TRADERS MASTERY RULES
# ============================================================================

TRADERS_MASTERY_EVALUATION = {
    # Tier: 5K
    5_000: TierRules(
        tier_name="5K Evaluation",
        min_balance=4_500,
        max_balance=5_500,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=5,
        max_lot_size=0.5,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 10K
    10_000: TierRules(
        tier_name="10K Evaluation",
        min_balance=9_000,
        max_balance=11_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=5,
        max_lot_size=1.0,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 25K
    25_000: TierRules(
        tier_name="25K Evaluation",
        min_balance=22_500,
        max_balance=27_500,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=10,
        max_lot_size=2.5,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 50K
    50_000: TierRules(
        tier_name="50K Evaluation",
        min_balance=45_000,
        max_balance=55_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=15,
        max_lot_size=5.0,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 100K
    100_000: TierRules(
        tier_name="100K Evaluation",
        min_balance=90_000,
        max_balance=110_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=20,
        max_lot_size=10.0,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 200K
    200_000: TierRules(
        tier_name="200K Evaluation",
        min_balance=180_000,
        max_balance=220_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=25,
        max_lot_size=20.0,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 300K
    300_000: TierRules(
        tier_name="300K Evaluation",
        min_balance=270_000,
        max_balance=330_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=30,
        max_lot_size=30.0,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    ),
}

TRADERS_MASTERY_ACCELERATED = {
    # Tier: 5K - Ускоренный режим (меньше цель, но быстрее)
    5_000: TierRules(
        tier_name="5K Accelerated",
        min_balance=4_500,
        max_balance=5_500,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=8.0,  # Жёстче
        profit_target_percent=5.0,       # Меньше цель
        max_leverage=50,                 # Меньше плечо
        max_positions=3,
        max_lot_size=0.3,
        min_trading_days=3,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 10K
    10_000: TierRules(
        tier_name="10K Accelerated",
        min_balance=9_000,
        max_balance=11_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=8.0,
        profit_target_percent=5.0,
        max_leverage=50,
        max_positions=5,
        max_lot_size=0.5,
        min_trading_days=3,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 25K
    25_000: TierRules(
        tier_name="25K Accelerated",
        min_balance=22_500,
        max_balance=27_500,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=8.0,
        profit_target_percent=5.0,
        max_leverage=50,
        max_positions=8,
        max_lot_size=1.25,
        min_trading_days=3,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 50K
    50_000: TierRules(
        tier_name="50K Accelerated",
        min_balance=45_000,
        max_balance=55_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=8.0,
        profit_target_percent=5.0,
        max_leverage=50,
        max_positions=10,
        max_lot_size=2.5,
        min_trading_days=3,
        weekend_holding=False,
        news_trading=False
    ),
    
    # Tier: 100K
    100_000: TierRules(
        tier_name="100K Accelerated",
        min_balance=90_000,
        max_balance=110_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=8.0,
        profit_target_percent=5.0,
        max_leverage=50,
        max_positions=15,
        max_lot_size=5.0,
        min_trading_days=3,
        weekend_holding=False,
        news_trading=False
    ),
}

TRADERS_MASTERY_MASTER = {
    # Funded аккаунты - без цели прибыли, только защита капитала
    
    # Tier: 5K
    5_000: TierRules(
        tier_name="5K Master (Funded)",
        min_balance=4_000,
        max_balance=6_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,  # Нет цели
        max_leverage=100,
        max_positions=5,
        max_lot_size=0.5,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 10K
    10_000: TierRules(
        tier_name="10K Master (Funded)",
        min_balance=8_000,
        max_balance=12_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=5,
        max_lot_size=1.0,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 25K
    25_000: TierRules(
        tier_name="25K Master (Funded)",
        min_balance=20_000,
        max_balance=30_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=10,
        max_lot_size=2.5,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 50K
    50_000: TierRules(
        tier_name="50K Master (Funded)",
        min_balance=40_000,
        max_balance=60_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=15,
        max_lot_size=5.0,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 100K
    100_000: TierRules(
        tier_name="100K Master (Funded)",
        min_balance=80_000,
        max_balance=120_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=20,
        max_lot_size=10.0,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 200K
    200_000: TierRules(
        tier_name="200K Master (Funded)",
        min_balance=160_000,
        max_balance=240_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=25,
        max_lot_size=20.0,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
    
    # Tier: 300K
    300_000: TierRules(
        tier_name="300K Master (Funded)",
        min_balance=240_000,
        max_balance=360_000,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=0.0,
        max_leverage=100,
        max_positions=30,
        max_lot_size=30.0,
        min_trading_days=0,
        weekend_holding=True,
        news_trading=True
    ),
}


# ============================================================================
# PROP FIRMS REGISTRY
# ============================================================================

PROP_FIRMS = {
    "traders_mastery": {
        "name": "Traders Mastery",
        "url": "https://tradersmastery.com",
        "platform": "tradelocker",
        "account_types": {
            AccountType.EVALUATION: TRADERS_MASTERY_EVALUATION,
            AccountType.ACCELERATED: TRADERS_MASTERY_ACCELERATED,
            AccountType.MASTER: TRADERS_MASTERY_MASTER,
        }
    },
    
    # Можно добавить другие проп-компании
    # "ftmo": {...},
    # "the5ers": {...},
}


def get_tier_rules(
    firm: str,
    account_type: AccountType,
    balance: float
) -> Optional[TierRules]:
    """
    Получение правил для конкретного баланса
    
    Args:
        firm: Название проп-компании (например "traders_mastery")
        account_type: Тип аккаунта
        balance: Текущий баланс
        
    Returns:
        TierRules или None если не найдено
    """
    if firm not in PROP_FIRMS:
        return None
    
    tiers = PROP_FIRMS[firm]["account_types"].get(account_type)
    if not tiers:
        return None
    
    # Ищем подходящий тир по балансу
    for tier_balance, rules in sorted(tiers.items()):
        if rules.min_balance <= balance <= rules.max_balance:
            return rules
    
    # Если не нашли точное совпадение - ищем ближайший
    closest_tier = min(tiers.keys(), key=lambda x: abs(x - balance))
    return tiers[closest_tier]


def detect_tier(balance: float, firm: str = "traders_mastery") -> tuple[int, TierRules]:
    """
    Автоматическое определение тира по балансу
    
    Args:
        balance: Баланс аккаунта
        firm: Проп-компания
        
    Returns:
        (tier_amount, TierRules)
    """
    # Стандартные тиры
    standard_tiers = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000]
    
    # Находим ближайший тир
    closest = min(standard_tiers, key=lambda x: abs(x - balance))
    
    # Пробуем разные типы аккаунтов в порядке приоритета
    for account_type in [AccountType.MASTER, AccountType.EVALUATION, AccountType.ACCELERATED]:
        rules = get_tier_rules(firm, account_type, balance)
        if rules:
            return closest, rules
    
    # Дефолтные правила если ничего не найдено
    return closest, TierRules(
        tier_name=f"{closest/1000:.0f}K Unknown",
        min_balance=closest * 0.9,
        max_balance=closest * 1.1,
        max_daily_loss_percent=5.0,
        max_total_drawdown_percent=10.0,
        profit_target_percent=8.0,
        max_leverage=100,
        max_positions=10,
        max_lot_size=closest / 10000,
        min_trading_days=5,
        weekend_holding=False,
        news_trading=False
    )

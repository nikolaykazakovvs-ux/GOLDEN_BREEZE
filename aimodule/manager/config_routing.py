"""
🎯 Golden Breeze - Execution Router Configuration
==================================================

Карта маршрутизации сигналов от единого AI Brain на конкретные торговые счета.

Архитектура:
- Один сигнал (например, "BTC UP") может быть исполнен на нескольких счетах
- Каждый счет имеет свой профиль риска (спот vs маржа)
- Router выбирает нужный счет и оборудование на основе класса ассета

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Тип торгового счета."""
    SPOT = "spot"              # Спотовый (MT5, MEXC spot)
    MARGIN = "margin"          # Маржа (TradeLocker, MT5 margin)
    PROP_FIRM = "prop_firm"   # Proprietary firm account


class RiskProfile(Enum):
    """Профиль риска для счета."""
    CONSERVATIVE = "conservative"    # 0.5% на сделку
    BALANCED = "balanced"            # 1% на сделку
    AGGRESSIVE = "aggressive"        # 2% на сделку
    FIXED = "fixed"                  # Фиксированный размер


@dataclass
class RiskConfig:
    """Конфигурация риска для счета."""
    profile: RiskProfile
    max_risk_percent: float = 1.0      # % от equity
    fixed_amount: Optional[float] = None  # Фиксированная сумма (USD)
    max_position_size: Optional[float] = None  # Макс размер позиции
    stop_loss_pips: Optional[float] = None    # SL в пипсах


@dataclass
class Account:
    """Описание торгового счета."""
    name: str                          # Уникальное имя счета (e.g., 'mexc_main')
    connector_type: str                # Тип коннектора ('MT5', 'MEXC', 'TRADELOCKER')
    account_type: AccountType          # SPOT / MARGIN / PROP_FIRM
    enabled: bool = True
    risk_config: RiskConfig = field(default_factory=lambda: RiskConfig(RiskProfile.BALANCED))
    metadata: Dict = field(default_factory=dict)  # Доп инфо (credentials, endpoints и т.д.)


@dataclass
class ExecutionTarget:
    """Конкретная цель исполнения для ордера."""
    account: Account           # Счет для исполнения
    symbol: str               # Торговая пара на этом счете (e.g., 'BTC/USDT')
    order_type: Literal['market', 'limit'] = 'market'
    max_slippage_percent: float = 0.5  # Макс проскальзывание
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# КОНФИГУРАЦИЯ СЧЕТОВ
# ============================================================================

ACCOUNTS: Dict[str, Account] = {
    # ========== MEXC - Спотовая торговля (Криптовалюты) ==========
    "mexc_spot_main": Account(
        name="mexc_spot_main",
        connector_type="MEXC",
        account_type=AccountType.SPOT,
        enabled=True,
        risk_config=RiskConfig(
            profile=RiskProfile.BALANCED,
            max_risk_percent=1.0,
            max_position_size=500.0  # USD
        ),
        metadata={
            "api_key": None,  # Заполнить из переменных окружения
            "api_secret": None,
            "description": "Main spot trading account for crypto"
        }
    ),

    # ========== MT5 - Форекс / Золото / Демо ==========
    "mt5_demo_xau": Account(
        name="mt5_demo_xau",
        connector_type="MT5",
        account_type=AccountType.MARGIN,
        enabled=True,
        risk_config=RiskConfig(
            profile=RiskProfile.CONSERVATIVE,
            max_risk_percent=0.5,
            max_position_size=50.0  # Лоты (не USD)
        ),
        metadata={
            "login": None,
            "password": None,
            "server": None,
            "description": "Demo MT5 account for Gold (XAUUSD)"
        }
    ),

    # ========== TradeLocker - Proprietary Firm Trading ==========
    "tradelocker_prop_1": Account(
        name="tradelocker_prop_1",
        connector_type="TRADELOCKER",
        account_type=AccountType.PROP_FIRM,
        enabled=True,
        risk_config=RiskConfig(
            profile=RiskProfile.AGGRESSIVE,
            max_risk_percent=2.0,
            max_position_size=1000.0  # USD
        ),
        metadata={
            "api_key": None,
            "api_secret": None,
            "description": "Proprietary account for speculative trading"
        }
    ),
}


# ============================================================================
# КАРТА МАРШРУТИЗАЦИИ (Asset Class -> Execution Targets)
# ============================================================================

ROUTING_MAP: Dict[str, List[ExecutionTarget]] = {
    # ========== БИТКОИН ==========
    "BTC": [
        # Первая цель: спотовая покупка на MEXC
        ExecutionTarget(
            account=ACCOUNTS["mexc_spot_main"],
            symbol="BTC/USDT",
            order_type="market",
            metadata={"purpose": "spot_accumulation"}
        ),
        # Вторая цель: спекуляция на TradeLocker (если уверенность > 75%)
        ExecutionTarget(
            account=ACCOUNTS["tradelocker_prop_1"],
            symbol="BTCUSD",
            order_type="market",
            metadata={"purpose": "speculative_leverage", "min_confidence": 0.75}
        ),
    ],

    # ========== ЭФИРИУМ ==========
    "ETH": [
        ExecutionTarget(
            account=ACCOUNTS["mexc_spot_main"],
            symbol="ETH/USDT",
            order_type="market",
            metadata={"purpose": "spot_accumulation"}
        ),
    ],

    # ========== ЗОЛОТО (XAUUSD) ==========
    "GOLD": [
        # MT5 демо
        ExecutionTarget(
            account=ACCOUNTS["mt5_demo_xau"],
            symbol="XAUUSD",
            order_type="market",
            metadata={"purpose": "forex_speculation"}
        ),
    ],

    # ========== ЕВРО (EURUSD) ==========
    "EUR": [
        ExecutionTarget(
            account=ACCOUNTS["mt5_demo_xau"],
            symbol="EURUSD",
            order_type="market",
            metadata={"purpose": "forex_speculation"}
        ),
    ],
}


# ============================================================================
# ПРАВИЛА ФИЛЬТРАЦИИ И УПРАВЛЕНИЯ СИГНАЛАМИ
# ============================================================================

SIGNAL_FILTER_RULES = {
    # Минимальная уверенность для каждого класса ассета
    "min_confidence": {
        "BTC": 0.55,
        "ETH": 0.60,
        "GOLD": 0.50,
        "EUR": 0.55,
    },

    # Максимальное количество открытых позиций на ассет
    "max_positions_per_asset": {
        "BTC": 2,      # Макс 2 позиции на BTC (спот + маржа)
        "ETH": 1,
        "GOLD": 1,
        "EUR": 1,
    },

    # Минимальный интервал между сигналами (сек)
    "min_signal_interval": {
        "BTC": 300,     # Минимум 5 минут между сигналами
        "ETH": 300,
        "GOLD": 300,
        "EUR": 300,
    },

    # Временные окна торговли (UTC)
    "trading_hours": {
        "BTC": {"start": "00:00", "end": "23:59"},  # 24/7
        "ETH": {"start": "00:00", "end": "23:59"},  # 24/7
        "GOLD": {"start": "01:00", "end": "22:00"},  # Во время сессии Лондон-Нью-Йорк
        "EUR": {"start": "08:00", "end": "22:00"},   # Европейская сессия
    },
}


# ============================================================================
# ФУНКЦИИ КОНФИГУРАЦИИ
# ============================================================================

def get_execution_targets(asset_class: str) -> List[ExecutionTarget]:
    """
    Получить все цели исполнения для конкретного класса ассета.

    Args:
        asset_class: Класс ассета (BTC, GOLD и т.д.)

    Returns:
        Список targets для исполнения
    """
    return ROUTING_MAP.get(asset_class, [])


def get_account(account_name: str) -> Optional[Account]:
    """Получить описание счета по имени."""
    return ACCOUNTS.get(account_name)


def get_enabled_accounts() -> List[Account]:
    """Получить все включённые счета."""
    return [acc for acc in ACCOUNTS.values() if acc.enabled]


def log_routing_config():
    """Вывести конфигурацию маршрутизации в лог."""
    logger.info("=" * 70)
    logger.info("EXECUTION ROUTING CONFIGURATION")
    logger.info("=" * 70)

    logger.info("\nACCOUNTS:")
    for name, account in ACCOUNTS.items():
        status = "✓ ENABLED" if account.enabled else "✗ DISABLED"
        logger.info(
            f"  • {name}: {account.connector_type} ({account.account_type.value}) [{status}]"
        )

    logger.info("\nROUTING MAP:")
    for asset, targets in ROUTING_MAP.items():
        logger.info(f"  {asset}:")
        for target in targets:
            logger.info(f"    → {target.account.name} : {target.symbol}")

    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_routing_config()

"""
🎯 Golden Breeze - Manager Module
==================================

Слой управления и маршрутизации торговых сигналов.

Структура:
- config_routing.py: Карта маршрутизации (какой сигнал куда идёт)
- trade_router.py: Маршрутизатор (умный распределитель)
- omni_loop.py: Главный цикл (Omniverse)

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

from .config_routing import (
    Account,
    AccountType,
    RiskProfile,
    RiskConfig,
    ExecutionTarget,
    ACCOUNTS,
    ROUTING_MAP,
    SIGNAL_FILTER_RULES,
    get_execution_targets,
    get_account,
    get_enabled_accounts,
    log_routing_config,
)

from .trade_router import (
    TradeRouter,
    AISignal,
    SignalDirection,
    ExecutionResult,
)

from .omni_loop import (
    OmniverseLoop,
)

__all__ = [
    # Config
    'Account',
    'AccountType',
    'RiskProfile',
    'RiskConfig',
    'ExecutionTarget',
    'ACCOUNTS',
    'ROUTING_MAP',
    'SIGNAL_FILTER_RULES',
    'get_execution_targets',
    'get_account',
    'get_enabled_accounts',
    'log_routing_config',
    
    # Router
    'TradeRouter',
    'AISignal',
    'SignalDirection',
    'ExecutionResult',
    
    # Omniverse
    'OmniverseLoop',
]

__version__ = '1.0.0'
__author__ = 'Golden Breeze Team'

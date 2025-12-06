"""
🎯 Golden Breeze - Trade Router (Order Execution Manager)
==========================================================

Умный распределитель торговых сигналов на конкретные счета и коннекторы.

Функционал:
- Получает сигнал от AI Brain ("BTC UP, confidence 85%")
- Смотрит в ROUTING_MAP и находит все возможные targets
- Калькулирует риск и размер позиции для каждого target
- Параллельно отправляет ордеры на все счета
- Обрабатывает ошибки индивидуально (если один счет упал - остальные продолжают)

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum

from .config_routing import (
    get_execution_targets,
    get_account,
    SIGNAL_FILTER_RULES,
    ExecutionTarget,
    Account,
)
from aimodule.connector.base import OrderSide, OrderType, OrderResult

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Направление сигнала от AI."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class AISignal:
    """Сигнал от AI Brain."""
    asset_class: str            # 'BTC', 'GOLD' и т.д.
    direction: SignalDirection  # UP / DOWN / NEUTRAL
    confidence: float           # 0.0 - 1.0
    timestamp: datetime
    metadata: Dict = None       # Дополнительная информация

    @property
    def strength(self) -> str:
        """Строковое описание уверенности."""
        if self.confidence >= 0.80:
            return "STRONG"
        elif self.confidence >= 0.65:
            return "MODERATE"
        else:
            return "WEAK"


@dataclass
class ExecutionResult:
    """Результат исполнения сигнала."""
    signal: AISignal
    target: ExecutionTarget
    success: bool
    order_id: Optional[str] = None
    order_result: Optional[OrderResult] = None
    error_message: Optional[str] = None
    timestamp: datetime = None


class TradeRouter:
    """
    Маршрутизатор торговых сигналов.

    Ключевые функции:
    1. Валидирует сигнал (confidence, timing, trading hours)
    2. Получает execution targets из ROUTING_MAP
    3. Калькулирует размер позиции для каждого target
    4. Асинхронно отправляет ордеры
    5. Логирует результаты
    """

    def __init__(self, connectors_dict: Dict[str, object]):
        """
        Args:
            connectors_dict: Словарь инициализированных коннекторов
                           {'MT5': mt5_connector, 'MEXC': mexc_connector, ...}
        """
        self.connectors = connectors_dict
        self.execution_history: List[ExecutionResult] = []
        self.signal_timestamps: Dict[str, datetime] = {}  # Отслеживание последнего сигнала
        
        logger.info("TradeRouter initialized with connectors:")
        for conn_type, conn in connectors_dict.items():
            logger.info(f"  • {conn_type}: {conn.__class__.__name__}")

    def validate_signal(self, signal: AISignal) -> Tuple[bool, Optional[str]]:
        """
        Валидирует сигнал перед исполнением.

        Returns:
            (is_valid, error_message)
        """
        # 1. Проверка уверенности
        min_conf = SIGNAL_FILTER_RULES["min_confidence"].get(signal.asset_class)
        if min_conf and signal.confidence < min_conf:
            return False, f"Confidence {signal.confidence:.2%} below minimum {min_conf:.2%}"

        # 2. Проверка направления (NEUTRAL не исполняем)
        if signal.direction == SignalDirection.NEUTRAL:
            return False, "NEUTRAL signals are not executed"

        # 3. Проверка минимального интервала между сигналами
        last_signal_time = self.signal_timestamps.get(signal.asset_class)
        if last_signal_time:
            interval = (signal.timestamp - last_signal_time).total_seconds()
            min_interval = SIGNAL_FILTER_RULES["min_signal_interval"].get(signal.asset_class, 300)
            if interval < min_interval:
                return False, f"Too soon (interval: {interval:.0f}s < {min_interval}s)"

        return True, None

    def calculate_position_size(
        self,
        target: ExecutionTarget,
        account_balance: float,
        asset_price: float,
        signal_confidence: float
    ) -> float:
        """
        Рассчитывает размер позиции для конкретного target.

        Args:
            target: Цель исполнения
            account_balance: Баланс счета (USD)
            asset_price: Текущая цена ассета
            signal_confidence: Уверенность сигнала (0-1)

        Returns:
            Размер позиции (в базовых единицах)
        """
        risk_config = target.account.risk_config

        # Если зафиксированный размер - используем его
        if risk_config.fixed_amount:
            position_size = risk_config.fixed_amount / asset_price
        else:
            # Калькулируем на основе % от equity
            risk_amount = account_balance * risk_config.max_risk_percent / 100.0

            # Корректируем на уверенность (более уверенные сигналы -> больше размер)
            if signal_confidence < 0.65:
                risk_amount *= 0.5  # 50% от базовой суммы
            elif signal_confidence > 0.80:
                risk_amount *= 1.0  # 100% от базовой суммы

            position_size = risk_amount / asset_price

        # Применяем макс размер позиции, если установлен
        if risk_config.max_position_size:
            position_size = min(position_size, risk_config.max_position_size / asset_price)

        return max(position_size, 0.0001)  # Минимум 0.0001

    async def execute_signal(
        self,
        signal: AISignal,
        account_balances: Optional[Dict[str, float]] = None,
        asset_prices: Optional[Dict[str, float]] = None
    ) -> List[ExecutionResult]:
        """
        Исполняет AI сигнал на все подходящие targets.

        Args:
            signal: AI сигнал
            account_balances: Баланс для каждого счета {account_name: balance}
            asset_prices: Цены ассетов {asset_class: price}

        Returns:
            Список результатов исполнения
        """
        logger.info(
            f"\n{'='*70}\n"
            f"EXECUTING SIGNAL: {signal.asset_class} {signal.direction.value.upper()} "
            f"({signal.confidence:.1%} confidence)\n"
            f"{'='*70}"
        )

        # 1. Валидируем сигнал
        is_valid, error = self.validate_signal(signal)
        if not is_valid:
            logger.warning(f"Signal validation failed: {error}")
            return []

        # 2. Обновляем timestamp последнего сигнала
        self.signal_timestamps[signal.asset_class] = signal.timestamp

        # 3. Получаем execution targets
        targets = get_execution_targets(signal.asset_class)
        if not targets:
            logger.warning(f"No execution targets found for {signal.asset_class}")
            return []

        logger.info(f"Found {len(targets)} execution targets")

        # 4. Асинхронно исполняем на все targets
        results = []
        tasks = []

        for target in targets:
            task = self._execute_order(
                signal=signal,
                target=target,
                account_balance=account_balances.get(target.account.name, 0.0) if account_balances else 0.0,
                asset_price=asset_prices.get(signal.asset_class, 0.0) if asset_prices else 0.0
            )
            tasks.append(task)

        # Ждём всех результатов (non-blocking)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 5. Логируем итоги
        successful = sum(1 for r in results if isinstance(r, ExecutionResult) and r.success)
        failed = len(results) - successful
        logger.info(
            f"\nExecution Summary: {successful}/{len(results)} successful, {failed} failed"
        )

        self.execution_history.extend([r for r in results if isinstance(r, ExecutionResult)])
        return [r for r in results if isinstance(r, ExecutionResult)]

    async def _execute_order(
        self,
        signal: AISignal,
        target: ExecutionTarget,
        account_balance: float,
        asset_price: float
    ) -> ExecutionResult:
        """
        Отправляет один ордер на конкретный target.

        Args:
            signal: AI сигнал
            target: Конкретная цель исполнения
            account_balance: Баланс счета
            asset_price: Цена ассета

        Returns:
            ExecutionResult с результатом исполнения
        """
        try:
            logger.info(
                f"\n→ Executing on {target.account.name} ({target.symbol})..."
            )

            # 1. Получаем коннектор
            connector = self.connectors.get(target.account.connector_type)
            if not connector:
                raise ValueError(f"Connector not found: {target.account.connector_type}")

            # 2. Калькулируем размер позиции
            volume = self.calculate_position_size(
                target=target,
                account_balance=account_balance,
                asset_price=asset_price,
                signal_confidence=signal.confidence
            )

            # 3. Определяем сторону (BUY/SELL)
            side = OrderSide.BUY if signal.direction == SignalDirection.UP else OrderSide.SELL

            # 4. Отправляем ордер
            order_result = await self._place_order(
                connector=connector,
                symbol=target.symbol,
                side=side,
                volume=volume,
                order_type=target.order_type
            )

            # 5. Формируем результат
            result = ExecutionResult(
                signal=signal,
                target=target,
                success=order_result is not None,
                order_id=order_result.order_id if order_result else None,
                order_result=order_result,
                timestamp=datetime.now()
            )

            if result.success:
                logger.info(
                    f"  ✓ Order placed: ID={result.order_id}, "
                    f"Volume={volume:.4f} {target.symbol}"
                )
            else:
                logger.error(f"  ✗ Order placement failed")

            return result

        except Exception as e:
            logger.error(f"  ✗ Error on {target.account.name}: {str(e)}", exc_info=True)
            return ExecutionResult(
                signal=signal,
                target=target,
                success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )

    async def _place_order(
        self,
        connector,
        symbol: str,
        side: OrderSide,
        volume: float,
        order_type: str
    ) -> Optional[OrderResult]:
        """
        Обёртка для размещения ордера (может быть асинхронной).
        """
        try:
            # Вызываем метод коннектора (может быть синхронным)
            if hasattr(connector, 'place_order_async'):
                return await connector.place_order_async(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    order_type=OrderType.MARKET if order_type == 'market' else OrderType.LIMIT
                )
            else:
                # Обычный синхронный вызов в async контексте
                return connector.place_order(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    order_type=OrderType.MARKET if order_type == 'market' else OrderType.LIMIT
                )
        except Exception as e:
            logger.error(f"Order placement error: {str(e)}")
            return None

    def get_execution_history(self, asset_class: Optional[str] = None) -> List[ExecutionResult]:
        """
        Получить историю исполнений.

        Args:
            asset_class: Если задан - только для этого ассета

        Returns:
            Список результатов исполнения
        """
        if asset_class:
            return [r for r in self.execution_history if r.signal.asset_class == asset_class]
        return self.execution_history

    def log_summary(self):
        """Вывести сводку по всем исполнениям."""
        logger.info("\n" + "=" * 70)
        logger.info("EXECUTION HISTORY SUMMARY")
        logger.info("=" * 70)

        if not self.execution_history:
            logger.info("No executions yet")
            return

        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        failed = total - successful

        logger.info(f"Total: {total}, Successful: {successful}, Failed: {failed}")
        logger.info("\nRecent executions:")

        for result in self.execution_history[-10:]:
            status = "✓" if result.success else "✗"
            logger.info(
                f"  {status} {result.signal.asset_class} {result.signal.direction.value.upper()} "
                f"@ {result.target.account.name} ({result.signal.confidence:.1%})"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("TradeRouter module loaded")

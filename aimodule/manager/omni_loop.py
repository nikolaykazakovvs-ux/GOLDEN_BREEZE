"""
🌍 Golden Breeze - Omniverse (Unified Trading Loop)
====================================================

Главный цикл торговой системы.

Архитектура:
1. Инициализирует ВСЕ коннекторы (MT5, MEXC, TradeLocker)
2. Параллельно собирает данные со всех источников
3. Кормит единый AI Brain (v5_ultimate)
4. Получает сигналы и отправляет на TradeRouter
5. Асинхронно исполняет на всех счетах

Цикл синхронизирован с M5 свечами (выполняется каждые 5 минут).

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from .config_routing import log_routing_config, Account, get_enabled_accounts
from .trade_router import TradeRouter, AISignal, SignalDirection
from aimodule.connector.mt5 import MT5Connector
from aimodule.connector.mexc import MEXCConnector
from aimodule.connector.tradelocker import TradeLockerConnector
from aimodule.inference.predict_direction import predict_direction
from aimodule.inference.combine_signals import combine_signals

logger = logging.getLogger(__name__)


class OmniverseLoop:
    """
    Главный цикл торговой системы.

    Ответственность:
    - Управление всеми коннекторами
    - Сбор данных со всех источников
    - Инфerence через AI модель
    - Маршрутизация на Trade Router
    - Синхронизация со временем (M5 свечи)
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        enable_mt5: bool = True,
        enable_mexc: bool = True,
        enable_tradelocker: bool = True,
        live_trading: bool = False
    ):
        """
        Args:
            config_file: Путь к JSON файлу с конфигурацией
            enable_mt5: Использовать MT5
            enable_mexc: Использовать MEXC
            enable_tradelocker: Использовать TradeLocker
            live_trading: Реальная торговля (True) или демо (False)
        """
        self.config_file = config_file
        self.live_trading = live_trading
        self.connectors: Dict[str, object] = {}
        self.router: Optional[TradeRouter] = None
        self.enabled_features = {
            'mt5': enable_mt5,
            'mexc': enable_mexc,
            'tradelocker': enable_tradelocker,
        }

        self.running = False
        self.stats = {
            'signals_processed': 0,
            'orders_executed': 0,
            'errors': 0,
            'loop_iterations': 0,
            'start_time': None,
        }

        logger.info("OmniverseLoop initialized")

    async def initialize(self) -> bool:
        """
        Инициализирует все коннекторы.

        Returns:
            True если успешно, False если ошибка
        """
        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING OMNIVERSE SYSTEM")
        logger.info("=" * 70)

        try:
            # 1. Выводим конфигурацию маршрутизации
            log_routing_config()

            # 2. Инициализируем коннекторы
            await self._initialize_connectors()

            # 3. Инициализируем Router
            self.router = TradeRouter(self.connectors)

            logger.info("\n✓ OMNIVERSE SYSTEM READY")
            logger.info("=" * 70 + "\n")
            return True

        except Exception as e:
            logger.error(f"✗ Initialization failed: {str(e)}", exc_info=True)
            return False

    async def _initialize_connectors(self):
        """Инициализирует все активные коннекторы."""
        logger.info("\nInitializing connectors...")

        # MT5
        if self.enabled_features['mt5']:
            try:
                logger.info("  • Connecting to MT5...")
                mt5 = MT5Connector()
                if mt5.connect():
                    self.connectors['MT5'] = mt5
                    logger.info("    ✓ MT5 connected")
                else:
                    logger.warning("    ✗ MT5 connection failed")
            except Exception as e:
                logger.warning(f"    ✗ MT5 error: {str(e)}")

        # MEXC
        if self.enabled_features['mexc']:
            try:
                logger.info("  • Connecting to MEXC...")
                mexc = MEXCConnector()
                if mexc.connect():
                    self.connectors['MEXC'] = mexc
                    logger.info("    ✓ MEXC connected")
                else:
                    logger.warning("    ✗ MEXC connection failed")
            except Exception as e:
                logger.warning(f"    ✗ MEXC error: {str(e)}")

        # TradeLocker
        if self.enabled_features['tradelocker']:
            try:
                logger.info("  • Connecting to TradeLocker...")
                tl = TradeLockerConnector()
                if tl.connect():
                    self.connectors['TRADELOCKER'] = tl
                    logger.info("    ✓ TradeLocker connected")
                else:
                    logger.warning("    ✗ TradeLocker connection failed")
            except Exception as e:
                logger.warning(f"    ✗ TradeLocker error: {str(e)}")

        if not self.connectors:
            raise RuntimeError("No connectors initialized!")

        logger.info(f"\nConnectors ready: {list(self.connectors.keys())}")

    async def collect_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Параллельно собирает данные со всех источников.

        Returns:
            Словарь {symbol: DataFrame} с данными OHLC
        """
        logger.info("\n[DATA COLLECTION] Gathering market data from all sources...")

        data = {}
        tasks = []

        # MT5: XAUUSD, EURUSD
        if 'MT5' in self.connectors:
            for symbol in ['XAUUSD', 'EURUSD']:
                task = self._fetch_symbol_data('MT5', symbol, 'M5', 200)
                tasks.append((symbol, task))

        # MEXC: BTC/USDT, ETH/USDT
        if 'MEXC' in self.connectors:
            for symbol in ['BTC/USDT', 'ETH/USDT']:
                task = self._fetch_symbol_data('MEXC', symbol, 'M5', 200)
                tasks.append((symbol, task))

        # Ждём всех одновременно
        results = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )

        for (symbol, _), result in zip(tasks, results):
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                data[symbol] = result
                logger.info(f"  ✓ {symbol}: {len(result)} bars")
            else:
                logger.warning(f"  ✗ {symbol}: Failed to collect")

        return data

    async def _fetch_symbol_data(
        self,
        connector_type: str,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> Optional[pd.DataFrame]:
        """Получает данные из одного коннектора."""
        try:
            connector = self.connectors[connector_type]

            # Вычисляем дату начала
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=bars * 5)

            # Получаем историю
            df = connector.get_history(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            return df if df is not None and len(df) > 0 else None

        except Exception as e:
            logger.debug(f"Error fetching {symbol} from {connector_type}: {str(e)}")
            return None

    async def inference(self, market_data: Dict[str, pd.DataFrame]) -> List[AISignal]:
        """
        Запускает AI модель и генерирует сигналы.

        Args:
            market_data: Данные со всех источников

        Returns:
            Список AI сигналов
        """
        logger.info("\n[INFERENCE] Running AI Brain...")

        signals = []

        try:
            # 1. XAUUSD (Золото с MT5)
            if 'XAUUSD' in market_data:
                direction, confidence = await self._predict_asset(
                    data=market_data['XAUUSD'],
                    asset_class='GOLD'
                )
                if direction != SignalDirection.NEUTRAL:
                    signals.append(AISignal(
                        asset_class='GOLD',
                        direction=direction,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={'source': 'MT5', 'symbol': 'XAUUSD'}
                    ))

            # 2. BTC (с MEXC)
            if 'BTC/USDT' in market_data:
                direction, confidence = await self._predict_asset(
                    data=market_data['BTC/USDT'],
                    asset_class='BTC'
                )
                if direction != SignalDirection.NEUTRAL:
                    signals.append(AISignal(
                        asset_class='BTC',
                        direction=direction,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={'source': 'MEXC', 'symbol': 'BTC/USDT'}
                    ))

            # 3. ETH (с MEXC)
            if 'ETH/USDT' in market_data:
                direction, confidence = await self._predict_asset(
                    data=market_data['ETH/USDT'],
                    asset_class='ETH'
                )
                if direction != SignalDirection.NEUTRAL:
                    signals.append(AISignal(
                        asset_class='ETH',
                        direction=direction,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={'source': 'MEXC', 'symbol': 'ETH/USDT'}
                    ))

            # 4. EUR (с MT5)
            if 'EURUSD' in market_data:
                direction, confidence = await self._predict_asset(
                    data=market_data['EURUSD'],
                    asset_class='EUR'
                )
                if direction != SignalDirection.NEUTRAL:
                    signals.append(AISignal(
                        asset_class='EUR',
                        direction=direction,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        metadata={'source': 'MT5', 'symbol': 'EURUSD'}
                    ))

            logger.info(f"  Generated {len(signals)} signals")
            return signals

        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            return []

    async def _predict_asset(
        self,
        data: pd.DataFrame,
        asset_class: str
    ) -> tuple:
        """
        Предсказывает направление для одного ассета.

        Returns:
            (direction, confidence)
        """
        try:
            # Используем имеющуюся модель (v5_ultimate)
            # Здесь нужно интегрировать реальную модель
            # На данный момент - возвращаем placeholder
            
            if len(data) < 50:
                return SignalDirection.NEUTRAL, 0.0

            # TODO: Реальное предсказание
            # prediction = predict_direction(data, model=self.model_v5_ultimate)
            # return prediction['direction'], prediction['confidence']

            # Placeholder для тестирования
            import random
            direction = random.choice([SignalDirection.UP, SignalDirection.DOWN, SignalDirection.NEUTRAL])
            confidence = random.uniform(0.5, 0.95)

            return direction, confidence

        except Exception as e:
            logger.error(f"Prediction error for {asset_class}: {str(e)}")
            return SignalDirection.NEUTRAL, 0.0

    async def execute_signals(self, signals: List[AISignal]) -> int:
        """
        Отправляет сигналы на Router для исполнения.

        Args:
            signals: Список AI сигналов

        Returns:
            Количество успешно исполненных сигналов
        """
        if not signals or not self.router:
            return 0

        logger.info(f"\n[EXECUTION] Routing {len(signals)} signals...")

        executed = 0

        for signal in signals:
            results = await self.router.execute_signal(signal)
            if results and any(r.success for r in results):
                executed += 1

        return executed

    async def run_loop(self, max_iterations: Optional[int] = None):
        """
        Главный цикл торговой системы.

        Args:
            max_iterations: Максимум итераций (None = бесконечно)
        """
        if not await self.initialize():
            logger.error("Failed to initialize Omniverse")
            return

        self.running = True
        self.stats['start_time'] = datetime.now()

        logger.info("\n🚀 OMNIVERSE LOOP STARTED\n")

        iteration = 0

        try:
            while self.running and (max_iterations is None or iteration < max_iterations):
                iteration += 1
                self.stats['loop_iterations'] = iteration

                loop_start = datetime.now()

                try:
                    logger.info(f"\n[ITERATION {iteration}] {loop_start.strftime('%H:%M:%S')}")
                    logger.info("-" * 70)

                    # 1. Сбор данных
                    market_data = await self.collect_market_data()

                    # 2. Inference
                    signals = await self.inference(market_data)
                    self.stats['signals_processed'] += len(signals)

                    # 3. Исполнение
                    executed = await self.execute_signals(signals)
                    self.stats['orders_executed'] += executed

                    # 4. Логирование итерации
                    loop_time = (datetime.now() - loop_start).total_seconds()
                    logger.info(f"Iteration completed in {loop_time:.2f}s")

                except Exception as e:
                    logger.error(f"Loop error: {str(e)}", exc_info=True)
                    self.stats['errors'] += 1

                # 5. Синхронизация со временем (ждём следующей M5 свечи)
                await self._sync_to_next_candle()

        except KeyboardInterrupt:
            logger.info("\n⏹️ Omniverse loop interrupted by user")

        finally:
            await self.shutdown()

    async def _sync_to_next_candle(self, timeframe_minutes: int = 5):
        """Ждёт следующей свечи (M5 по умолчанию)."""
        now = datetime.now()
        minutes_since_hour = now.minute
        next_candle_minute = ((minutes_since_hour // timeframe_minutes) + 1) * timeframe_minutes

        if next_candle_minute >= 60:
            next_candle_minute = 0
            next_candle_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
        else:
            next_candle_time = now.replace(minute=next_candle_minute, second=0, microsecond=0)

        sleep_seconds = (next_candle_time - now).total_seconds()
        if sleep_seconds > 0:
            logger.info(f"Sleeping {sleep_seconds:.0f}s until next M5 candle...")
            await asyncio.sleep(sleep_seconds)

    async def shutdown(self):
        """Корректно завершает работу всех коннекторов."""
        logger.info("\n" + "=" * 70)
        logger.info("SHUTTING DOWN OMNIVERSE")
        logger.info("=" * 70)

        self.running = False

        # Закрываем коннекторы
        for conn_name, connector in self.connectors.items():
            try:
                if hasattr(connector, 'disconnect'):
                    connector.disconnect()
                logger.info(f"  ✓ {conn_name} disconnected")
            except Exception as e:
                logger.warning(f"  ✗ {conn_name} disconnect error: {str(e)}")

        # Выводим статистику
        self._print_stats()

    def _print_stats(self):
        """Выводит статистику работы системы."""
        if not self.stats['start_time']:
            return

        uptime = datetime.now() - self.stats['start_time']

        logger.info("\n" + "=" * 70)
        logger.info("OMNIVERSE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Uptime: {uptime}")
        logger.info(f"Loop Iterations: {self.stats['loop_iterations']}")
        logger.info(f"Signals Processed: {self.stats['signals_processed']}")
        logger.info(f"Orders Executed: {self.stats['orders_executed']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.router:
            self.router.log_summary()

        logger.info("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Главная функция для запуска Omniverse."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Создаём Omniverse
    omniverse = OmniverseLoop(
        enable_mt5=True,
        enable_mexc=True,
        enable_tradelocker=True,
        live_trading=False  # Демо режим
    )

    # Запускаем (5 итераций для тестирования)
    await omniverse.run_loop(max_iterations=5)


if __name__ == "__main__":
    asyncio.run(main())

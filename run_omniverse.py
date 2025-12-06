#!/usr/bin/env python3
"""
🌍 Golden Breeze - Omniverse Launch Script
===========================================

Запускает единую торговую систему.

Использование:
    python run_omniverse.py [--demo] [--live] [--iterations N]

Options:
    --demo              Демо режим (без реальных ордеров)
    --live              Реальная торговля
    --iterations N      Количество итераций (по умолчанию = бесконечно)

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

import sys
import asyncio
import logging
import argparse
from pathlib import Path

# Добавляем корневую директорию проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aimodule.manager import OmniverseLoop, log_routing_config

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


async def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Golden Breeze - Omniverse Trading System"
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo mode (no real orders)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Live trading mode'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of iterations'
    )
    parser.add_argument(
        '--no-mt5',
        action='store_true',
        help='Disable MT5 connector'
    )
    parser.add_argument(
        '--no-mexc',
        action='store_true',
        help='Disable MEXC connector'
    )
    parser.add_argument(
        '--no-tl',
        action='store_true',
        help='Disable TradeLocker connector'
    )

    args = parser.parse_args()

    # Проверяем режим
    if args.live and args.demo:
        logger.error("Cannot use both --live and --demo flags")
        sys.exit(1)

    live_trading = args.live or not args.demo

    logger.info("\n" + "=" * 70)
    logger.info("🌍 GOLDEN BREEZE - OMNIVERSE TRADING SYSTEM")
    logger.info("=" * 70)
    logger.info(f"Mode: {'LIVE' if live_trading else 'DEMO'}")
    logger.info(f"Iterations: {args.iterations or 'infinite'}")
    logger.info(f"MT5: {'enabled' if not args.no_mt5 else 'disabled'}")
    logger.info(f"MEXC: {'enabled' if not args.no_mexc else 'disabled'}")
    logger.info(f"TradeLocker: {'enabled' if not args.no_tl else 'disabled'}")
    logger.info("=" * 70 + "\n")

    # Создаём Omniverse
    omniverse = OmniverseLoop(
        enable_mt5=not args.no_mt5,
        enable_mexc=not args.no_mexc,
        enable_tradelocker=not args.no_tl,
        live_trading=live_trading
    )

    # Запускаем
    try:
        await omniverse.run_loop(max_iterations=args.iterations)
    except KeyboardInterrupt:
        logger.info("\nGraceful shutdown...")
        await omniverse.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

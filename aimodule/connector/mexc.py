"""
MEXC Crypto Exchange Connector
Коннектор для криптовалютной биржи MEXC через ccxt
"""

import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from .base import (
    BaseConnector,
    OrderSide,
    OrderType,
    OrderResult,
    Position,
    AccountInfo
)

logger = logging.getLogger(__name__)


# Маппинг таймфреймов в формат ccxt
TIMEFRAME_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
    "MN1": "1M",
    # Также поддержка прямого формата ccxt
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}


class MEXCConnector(BaseConnector):
    """
    Коннектор для биржи MEXC
    Поддерживает spot и futures торговлю
    """
    
    SOURCE_NAME = "mexc"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        market_type: str = "spot"  # "spot" или "futures"
    ):
        """
        Args:
            api_key: API ключ MEXC
            api_secret: API секрет MEXC
            testnet: Использовать тестовую сеть
            market_type: "spot" для спотовой торговли, "futures" для фьючерсов
        """
        super().__init__()
        
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt не установлен. Запустите: pip install ccxt")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.market_type = market_type
        
        self.exchange: Optional[ccxt.mexc] = None
        self._connected = False
        
    @property
    def is_connected(self) -> bool:
        return self._connected and self.exchange is not None
    
    def connect(self) -> bool:
        """Подключение к MEXC"""
        try:
            # Создаём exchange объект
            exchange_class = ccxt.mexc
            
            config = {
                'enableRateLimit': True,
                'rateLimit': 100,  # ms между запросами
            }
            
            if self.api_key and self.api_secret:
                config['apiKey'] = self.api_key
                config['secret'] = self.api_secret
            
            if self.testnet:
                config['sandbox'] = True
            
            if self.market_type == "futures":
                config['options'] = {
                    'defaultType': 'swap',  # для futures
                }
            
            self.exchange = exchange_class(config)
            
            # Загружаем рынки
            self.exchange.load_markets()
            
            self._connected = True
            logger.info(f"✅ MEXC {self.market_type} подключен. Доступно {len(self.exchange.markets)} рынков")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к MEXC: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """Отключение от MEXC"""
        self.exchange = None
        self._connected = False
        logger.info("MEXC отключен")
        return True
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 1000
    ) -> pd.DataFrame:
        """
        Получение исторических данных
        
        Args:
            symbol: Символ (например "BTC/USDT" или "BTCUSDT")
            timeframe: Таймфрейм (M1, M5, H1 или 1m, 5m, 1h)
            start_date: Начальная дата
            end_date: Конечная дата
            count: Количество баров
            
        Returns:
            DataFrame с OHLCV данными
        """
        if not self.is_connected:
            logger.error("MEXC не подключен")
            return pd.DataFrame()
        
        try:
            # Нормализуем символ (BTCUSDT -> BTC/USDT)
            symbol = self._normalize_symbol_mexc(symbol)
            
            # Конвертируем таймфрейм
            tf = TIMEFRAME_MAP.get(timeframe, timeframe)
            
            # Определяем since (время в миллисекундах)
            since = None
            if start_date:
                since = int(start_date.timestamp() * 1000)
            
            # Получаем OHLCV данные
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                since=since,
                limit=count
            )
            
            if not ohlcv:
                logger.warning(f"Нет данных для {symbol} {tf}")
                return pd.DataFrame()
            
            # Конвертируем в DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Конвертируем время из миллисекунд в datetime
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            
            # Фильтруем по end_date если указано
            if end_date:
                end_ts = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date
                df = df[df['time'] <= end_ts]
            
            # Сортируем по времени
            df = df.sort_values('time').reset_index(drop=True)
            
            # Добавляем tick_volume как копию volume (для совместимости)
            df['tick_volume'] = df['volume']
            df['spread'] = 0  # Спреда нет на крипто
            df['real_volume'] = df['volume']
            
            logger.info(f"📊 MEXC: получено {len(df)} баров {symbol} {tf}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории MEXC: {e}")
            return pd.DataFrame()
    
    def get_balance(self) -> float:
        """Получение баланса USDT"""
        if not self.is_connected:
            return 0.0
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Ищем USDT баланс
            if 'USDT' in balance:
                return float(balance['USDT'].get('free', 0) or 0)
            
            # Альтернативный путь
            if 'free' in balance and 'USDT' in balance['free']:
                return float(balance['free']['USDT'] or 0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Ошибка получения баланса MEXC: {e}")
            return 0.0
    
    def get_account_info(self) -> AccountInfo:
        """Получение информации об аккаунте"""
        if not self.is_connected:
            return AccountInfo(
                balance=0,
                equity=0,
                margin=0,
                free_margin=0,
                currency="USDT"
            )
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Считаем общий баланс в USDT
            total_usdt = float(balance.get('total', {}).get('USDT', 0) or 0)
            free_usdt = float(balance.get('free', {}).get('USDT', 0) or 0)
            used_usdt = float(balance.get('used', {}).get('USDT', 0) or 0)
            
            return AccountInfo(
                balance=total_usdt,
                equity=total_usdt,
                margin=used_usdt,
                free_margin=free_usdt,
                currency="USDT"
            )
            
        except Exception as e:
            logger.error(f"Ошибка получения аккаунта MEXC: {e}")
            return AccountInfo(
                balance=0,
                equity=0,
                margin=0,
                free_margin=0,
                currency="USDT"
            )
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = ""
    ) -> OrderResult:
        """Размещение ордера на MEXC"""
        if not self.is_connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="MEXC не подключен"
            )
        
        try:
            symbol = self._normalize_symbol_mexc(symbol)
            
            # Определяем тип ордера ccxt
            ccxt_type = 'market' if order_type == OrderType.MARKET else 'limit'
            ccxt_side = 'buy' if side == OrderSide.BUY else 'sell'
            
            # Параметры ордера
            params = {}
            if comment:
                params['clientOrderId'] = comment[:32]  # Ограничение MEXC
            
            # Для spot рынка
            if self.market_type == "spot":
                if order_type == OrderType.MARKET:
                    order = self.exchange.create_market_order(
                        symbol=symbol,
                        side=ccxt_side,
                        amount=volume,
                        params=params
                    )
                else:
                    order = self.exchange.create_limit_order(
                        symbol=symbol,
                        side=ccxt_side,
                        amount=volume,
                        price=price,
                        params=params
                    )
            else:
                # Для futures добавляем SL/TP если есть
                if sl:
                    params['stopLoss'] = {'triggerPrice': sl}
                if tp:
                    params['takeProfit'] = {'triggerPrice': tp}
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=ccxt_type,
                    side=ccxt_side,
                    amount=volume,
                    price=price,
                    params=params
                )
            
            order_id = order.get('id', str(order.get('info', {}).get('orderId', '')))
            
            logger.info(f"✅ MEXC ордер: {ccxt_side} {volume} {symbol}, ID={order_id}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                executed_price=float(order.get('average', 0) or order.get('price', 0) or 0),
                executed_volume=float(order.get('filled', volume) or volume),
                message="Ордер успешно размещён"
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения ордера MEXC: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=str(e)
            )
    
    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Получение открытых позиций"""
        if not self.is_connected:
            return []
        
        positions = []
        
        try:
            if self.market_type == "spot":
                # Для spot - смотрим баланс активов
                balance = self.exchange.fetch_balance()
                
                for asset, info in balance.get('total', {}).items():
                    if asset == 'USDT' or float(info or 0) == 0:
                        continue
                    
                    # Фильтруем по символу если указан
                    if symbol:
                        normalized = self._normalize_symbol_mexc(symbol)
                        if asset not in normalized:
                            continue
                    
                    # Получаем текущую цену
                    try:
                        ticker = self.exchange.fetch_ticker(f"{asset}/USDT")
                        current_price = float(ticker['last'])
                        volume = float(info)
                        
                        positions.append(Position(
                            symbol=f"{asset}/USDT",
                            side=OrderSide.BUY,
                            volume=volume,
                            open_price=current_price,  # Нет информации о цене входа
                            current_price=current_price,
                            profit=0,  # Нельзя рассчитать без цены входа
                            open_time=datetime.now(timezone.utc)
                        ))
                    except:
                        pass
            else:
                # Для futures - используем fetch_positions
                if symbol:
                    symbol = self._normalize_symbol_mexc(symbol)
                    raw_positions = self.exchange.fetch_positions([symbol])
                else:
                    raw_positions = self.exchange.fetch_positions()
                
                for pos in raw_positions:
                    if float(pos.get('contracts', 0) or 0) == 0:
                        continue
                    
                    side = OrderSide.BUY if pos['side'] == 'long' else OrderSide.SELL
                    
                    positions.append(Position(
                        symbol=pos['symbol'],
                        side=side,
                        volume=float(pos.get('contracts', 0)),
                        open_price=float(pos.get('entryPrice', 0) or 0),
                        current_price=float(pos.get('markPrice', 0) or 0),
                        profit=float(pos.get('unrealizedPnl', 0) or 0),
                        open_time=datetime.now(timezone.utc)  # MEXC не даёт время открытия
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Ошибка получения позиций MEXC: {e}")
            return []
    
    def close_position(
        self,
        symbol: str,
        volume: Optional[float] = None
    ) -> OrderResult:
        """Закрытие позиции"""
        if not self.is_connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="MEXC не подключен"
            )
        
        try:
            symbol = self._normalize_symbol_mexc(symbol)
            
            # Находим позицию
            positions = self.get_positions(symbol)
            if not positions:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message=f"Нет открытых позиций для {symbol}"
                )
            
            pos = positions[0]
            close_volume = volume or pos.volume
            
            # Для закрытия - противоположный ордер
            close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            
            return self.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                volume=close_volume,
                comment="close_position"
            )
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции MEXC: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=str(e)
            )
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены"""
        if not self.is_connected:
            return None
        
        try:
            symbol = self._normalize_symbol_mexc(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Ошибка получения цены MEXC: {e}")
            return None
    
    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Получение полной информации о тикере"""
        if not self.is_connected:
            return None
        
        try:
            symbol = self._normalize_symbol_mexc(symbol)
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Ошибка получения тикера MEXC: {e}")
            return None
    
    def get_available_symbols(self) -> list[str]:
        """Получение списка доступных символов"""
        if not self.is_connected:
            return []
        
        try:
            return list(self.exchange.markets.keys())
        except:
            return []
    
    def _normalize_symbol_mexc(self, symbol: str) -> str:
        """
        Нормализация символа в формат MEXC
        BTCUSDT -> BTC/USDT
        BTC/USDT -> BTC/USDT
        """
        symbol = symbol.upper().strip()
        
        # Уже в правильном формате
        if '/' in symbol:
            return symbol
        
        # Пытаемся разделить по известным quote currencies
        quote_currencies = ['USDT', 'USDC', 'BTC', 'ETH', 'BUSD', 'USD']
        
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:
                    return f"{base}/{quote}"
        
        # Не удалось разобрать - возвращаем как есть
        return symbol


# Псевдоним для удобства
MexcConnector = MEXCConnector

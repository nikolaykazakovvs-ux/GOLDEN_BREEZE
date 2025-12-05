"""
TradeLocker Connector
Коннектор для TradeLocker (проп-фирмы, фьючерсы)
Использует REST API с Token Authentication
"""

import logging
import hmac
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Any
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base import (
    BaseConnector,
    OrderSide,
    OrderType,
    OrderResult,
    Position,
    AccountInfo
)

logger = logging.getLogger(__name__)


class TradeLockerConnector(BaseConnector):
    """
    Коннектор для TradeLocker
    Поддерживает торговлю фьючерсами через проп-фирмы
    """
    
    SOURCE_NAME = "tradelocker"
    
    # API endpoints
    BASE_URL_LIVE = "https://live.tradelocker.com/backend-api"
    BASE_URL_DEMO = "https://demo.tradelocker.com/backend-api"
    
    # Маппинг таймфреймов
    TIMEFRAME_MAP = {
        "M1": "1",
        "M5": "5", 
        "M15": "15",
        "M30": "30",
        "H1": "60",
        "H4": "240",
        "D1": "1440",
        "W1": "10080",
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "1440",
    }
    
    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        account_id: Optional[str] = None,
        demo: bool = True
    ):
        """
        Args:
            email: Email для входа
            password: Пароль
            server: Сервер TradeLocker
            account_id: ID торгового аккаунта
            demo: True для демо, False для live
        """
        super().__init__()
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests не установлен. Запустите: pip install requests")
        
        self.email = email
        self.password = password
        self.server = server
        self.account_id = account_id
        self.demo = demo
        
        self.base_url = self.BASE_URL_DEMO if demo else self.BASE_URL_LIVE
        
        # Токены авторизации
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
        self._connected = False
        self._session = requests.Session()
        
        # Кэш символов
        self._instruments_cache: dict = {}
        
    @property
    def is_connected(self) -> bool:
        return self._connected and self.access_token is not None
    
    def connect(self) -> bool:
        """Подключение к TradeLocker"""
        try:
            # Шаг 1: Авторизация
            auth_response = self._authenticate()
            if not auth_response:
                return False
            
            # Шаг 2: Получаем аккаунты если account_id не указан
            if not self.account_id:
                accounts = self._get_accounts()
                if accounts:
                    self.account_id = accounts[0]['id']
                    logger.info(f"Выбран аккаунт: {self.account_id}")
                else:
                    logger.error("Нет доступных аккаунтов")
                    return False
            
            # Шаг 3: Загружаем инструменты
            self._load_instruments()
            
            self._connected = True
            logger.info(f"✅ TradeLocker подключен (demo={self.demo})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к TradeLocker: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """Отключение от TradeLocker"""
        try:
            # Можно вызвать logout endpoint если есть
            pass
        except:
            pass
        
        self.access_token = None
        self.refresh_token = None
        self._connected = False
        logger.info("TradeLocker отключен")
        return True
    
    def _authenticate(self) -> bool:
        """Авторизация и получение токенов"""
        try:
            url = f"{self.base_url}/auth/jwt/token"
            
            payload = {
                "email": self.email,
                "password": self.password,
                "server": self.server
            }
            
            response = self._session.post(url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Ошибка авторизации: {response.status_code} - {response.text}")
                return False
            
            data = response.json()
            
            self.access_token = data.get('accessToken')
            self.refresh_token = data.get('refreshToken')
            
            # Токен действует ~15 минут
            self.token_expires = datetime.now(timezone.utc) + timedelta(minutes=14)
            
            # Обновляем заголовки сессии
            self._session.headers.update({
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            })
            
            logger.info("✅ TradeLocker авторизация успешна")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            return False
    
    def _refresh_token_if_needed(self):
        """Обновление токена если истёк"""
        if not self.token_expires:
            return
        
        if datetime.now(timezone.utc) >= self.token_expires:
            self._refresh_access_token()
    
    def _refresh_access_token(self) -> bool:
        """Обновление access token"""
        try:
            url = f"{self.base_url}/auth/jwt/refresh"
            
            payload = {
                "refreshToken": self.refresh_token
            }
            
            response = self._session.post(url, json=payload)
            
            if response.status_code != 200:
                # Если refresh не работает - переавторизуемся
                return self._authenticate()
            
            data = response.json()
            
            self.access_token = data.get('accessToken')
            self.refresh_token = data.get('refreshToken', self.refresh_token)
            self.token_expires = datetime.now(timezone.utc) + timedelta(minutes=14)
            
            self._session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления токена: {e}")
            return False
    
    def _get_accounts(self) -> list[dict]:
        """Получение списка аккаунтов"""
        try:
            self._refresh_token_if_needed()
            
            url = f"{self.base_url}/auth/jwt/all-accounts"
            response = self._session.get(url)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            return data.get('accounts', [])
            
        except Exception as e:
            logger.error(f"Ошибка получения аккаунтов: {e}")
            return []
    
    def _load_instruments(self):
        """Загрузка списка инструментов"""
        try:
            self._refresh_token_if_needed()
            
            url = f"{self.base_url}/trade/instruments"
            
            headers = {
                'accNum': str(self.account_id)
            }
            
            response = self._session.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                instruments = data.get('d', {}).get('instruments', [])
                
                for inst in instruments:
                    symbol = inst.get('name', '')
                    self._instruments_cache[symbol] = inst
                
                logger.info(f"Загружено {len(self._instruments_cache)} инструментов")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки инструментов: {e}")
    
    def _api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None
    ) -> Optional[dict]:
        """Выполнение API запроса"""
        self._refresh_token_if_needed()
        
        url = f"{self.base_url}{endpoint}"
        
        headers = {}
        if self.account_id:
            headers['accNum'] = str(self.account_id)
        
        try:
            if method.upper() == 'GET':
                response = self._session.get(url, params=params, headers=headers)
            elif method.upper() == 'POST':
                response = self._session.post(url, json=json_data, headers=headers)
            elif method.upper() == 'DELETE':
                response = self._session.delete(url, headers=headers)
            else:
                return None
            
            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 1000
    ) -> pd.DataFrame:
        """Получение исторических данных"""
        if not self.is_connected:
            logger.error("TradeLocker не подключен")
            return pd.DataFrame()
        
        try:
            # Конвертируем таймфрейм
            tf_minutes = self.TIMEFRAME_MAP.get(timeframe, "60")
            
            # Определяем временные рамки
            end_ts = int((end_date or datetime.now(timezone.utc)).timestamp() * 1000)
            
            if start_date:
                start_ts = int(start_date.timestamp() * 1000)
            else:
                # По умолчанию - последние N баров
                minutes_per_bar = int(tf_minutes)
                start_ts = end_ts - (count * minutes_per_bar * 60 * 1000)
            
            # Получаем instrument ID
            instrument_id = self._get_instrument_id(symbol)
            if not instrument_id:
                logger.error(f"Инструмент {symbol} не найден")
                return pd.DataFrame()
            
            # Запрос истории
            endpoint = f"/trade/history/{instrument_id}/{tf_minutes}"
            params = {
                'from': start_ts,
                'to': end_ts
            }
            
            data = self._api_request('GET', endpoint, params=params)
            
            if not data or 'd' not in data:
                logger.warning(f"Нет данных для {symbol}")
                return pd.DataFrame()
            
            bars = data['d'].get('barData', [])
            if not bars:
                return pd.DataFrame()
            
            # Парсим данные
            records = []
            for bar in bars:
                records.append({
                    'time': datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc),
                    'open': float(bar[1]),
                    'high': float(bar[2]),
                    'low': float(bar[3]),
                    'close': float(bar[4]),
                    'volume': float(bar[5]) if len(bar) > 5 else 0
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('time').reset_index(drop=True)
            
            # Добавляем стандартные колонки
            df['tick_volume'] = df['volume']
            df['spread'] = 0
            df['real_volume'] = df['volume']
            
            logger.info(f"📊 TradeLocker: получено {len(df)} баров {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения истории TradeLocker: {e}")
            return pd.DataFrame()
    
    def _get_instrument_id(self, symbol: str) -> Optional[int]:
        """Получение ID инструмента по символу"""
        symbol = symbol.upper().strip()
        
        if symbol in self._instruments_cache:
            return self._instruments_cache[symbol].get('tradableInstrumentId')
        
        # Пробуем с разными вариациями
        variations = [symbol, symbol.replace('/', ''), f"{symbol}"]
        
        for var in variations:
            for name, inst in self._instruments_cache.items():
                if name.upper() == var or name.upper().replace('/', '') == var:
                    return inst.get('tradableInstrumentId')
        
        return None
    
    def get_balance(self) -> float:
        """Получение баланса"""
        info = self.get_account_info()
        return info.balance
    
    def get_account_info(self) -> AccountInfo:
        """Получение информации об аккаунте"""
        if not self.is_connected:
            return AccountInfo(
                balance=0,
                equity=0,
                margin=0,
                free_margin=0,
                currency="USD"
            )
        
        try:
            data = self._api_request('GET', '/trade/accounts')
            
            if not data or 'd' not in data:
                return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, currency="USD")
            
            accounts = data['d'].get('accounts', [])
            
            # Ищем текущий аккаунт
            for acc in accounts:
                if str(acc.get('id')) == str(self.account_id):
                    return AccountInfo(
                        balance=float(acc.get('balance', 0)),
                        equity=float(acc.get('equity', 0)),
                        margin=float(acc.get('usedMargin', 0)),
                        free_margin=float(acc.get('freeMargin', 0)),
                        currency=acc.get('currency', 'USD')
                    )
            
            return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, currency="USD")
            
        except Exception as e:
            logger.error(f"Ошибка получения аккаунта: {e}")
            return AccountInfo(balance=0, equity=0, margin=0, free_margin=0, currency="USD")
    
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
        """Размещение ордера"""
        if not self.is_connected:
            return OrderResult(
                success=False,
                order_id=None,
                message="TradeLocker не подключен"
            )
        
        try:
            instrument_id = self._get_instrument_id(symbol)
            if not instrument_id:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message=f"Инструмент {symbol} не найден"
                )
            
            # Формируем ордер
            order_data = {
                "tradableInstrumentId": instrument_id,
                "side": "buy" if side == OrderSide.BUY else "sell",
                "type": "market" if order_type == OrderType.MARKET else "limit",
                "qty": volume
            }
            
            if order_type == OrderType.LIMIT and price:
                order_data["price"] = price
            
            if sl:
                order_data["stopLoss"] = sl
            if tp:
                order_data["takeProfit"] = tp
            
            data = self._api_request('POST', '/trade/orders', json_data=order_data)
            
            if not data or 'd' not in data:
                return OrderResult(
                    success=False,
                    order_id=None,
                    message="Ошибка размещения ордера"
                )
            
            order_id = data['d'].get('orderId', '')
            
            logger.info(f"✅ TradeLocker ордер: {side.value} {volume} {symbol}, ID={order_id}")
            
            return OrderResult(
                success=True,
                order_id=str(order_id),
                message="Ордер успешно размещён"
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка размещения ордера TradeLocker: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                message=str(e)
            )
    
    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Получение открытых позиций"""
        if not self.is_connected:
            return []
        
        try:
            data = self._api_request('GET', '/trade/positions')
            
            if not data or 'd' not in data:
                return []
            
            positions = []
            raw_positions = data['d'].get('positions', [])
            
            for pos in raw_positions:
                pos_symbol = pos.get('symbol', '')
                
                # Фильтруем по символу если указан
                if symbol and symbol.upper() not in pos_symbol.upper():
                    continue
                
                side = OrderSide.BUY if pos.get('side') == 'buy' else OrderSide.SELL
                
                positions.append(Position(
                    symbol=pos_symbol,
                    side=side,
                    volume=float(pos.get('qty', 0)),
                    open_price=float(pos.get('avgPrice', 0)),
                    current_price=float(pos.get('currentPrice', 0)),
                    profit=float(pos.get('unrealizedPnl', 0)),
                    open_time=datetime.now(timezone.utc)
                ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Ошибка получения позиций: {e}")
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
                message="TradeLocker не подключен"
            )
        
        try:
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
            
            # Закрываем противоположным ордером
            close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
            
            return self.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                volume=close_volume,
                comment="close_position"
            )
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции: {e}")
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
            instrument_id = self._get_instrument_id(symbol)
            if not instrument_id:
                return None
            
            data = self._api_request('GET', f'/trade/quotes/{instrument_id}')
            
            if data and 'd' in data:
                bid = float(data['d'].get('bid', 0))
                ask = float(data['d'].get('ask', 0))
                return (bid + ask) / 2
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка получения цены: {e}")
            return None
    
    def get_available_symbols(self) -> list[str]:
        """Получение списка доступных символов"""
        return list(self._instruments_cache.keys())


# Псевдоним для удобства
TradlockerConnector = TradeLockerConnector

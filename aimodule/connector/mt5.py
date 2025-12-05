"""
🔌 Golden Breeze - MT5 Connector
================================

MetaTrader 5 connector implementation.
Supports Forex, Gold, and other MT5 instruments.

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import pandas as pd
import logging

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

from .base import (
    BaseConnector,
    OrderSide,
    OrderType,
    OrderResult,
    Position,
    AccountInfo,
)


logger = logging.getLogger(__name__)


class MT5Connector(BaseConnector):
    """
    MetaTrader 5 connector.
    
    Provides access to Forex, Gold, and other MT5 instruments.
    
    Usage:
        connector = MT5Connector()
        connector.connect()
        df = connector.get_history("XAUUSD", "M5", start, end)
    """
    
    PLATFORM_NAME = "MT5"
    SUPPORTED_SYMBOLS = {"XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"}
    SUPPORTED_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}
    
    # MT5 timeframe mapping
    TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1 if HAS_MT5 else 1,
        "M5": mt5.TIMEFRAME_M5 if HAS_MT5 else 5,
        "M15": mt5.TIMEFRAME_M15 if HAS_MT5 else 15,
        "M30": mt5.TIMEFRAME_M30 if HAS_MT5 else 30,
        "H1": mt5.TIMEFRAME_H1 if HAS_MT5 else 60,
        "H4": mt5.TIMEFRAME_H4 if HAS_MT5 else 240,
        "D1": mt5.TIMEFRAME_D1 if HAS_MT5 else 1440,
        "W1": mt5.TIMEFRAME_W1 if HAS_MT5 else 10080,
    }
    
    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MT5 connector.
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            path: Path to MT5 terminal
            credentials: Alternative dict with all credentials
        """
        super().__init__(credentials)
        
        # Extract credentials
        if credentials:
            self.login = credentials.get('login', login)
            self.password = credentials.get('password', password)
            self.server = credentials.get('server', server)
            self.path = credentials.get('path', path)
        else:
            self.login = login
            self.password = password
            self.server = server
            self.path = path
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not HAS_MT5:
            self._last_error = "MetaTrader5 package not installed"
            logger.error(self._last_error)
            return False
        
        logger.info("🔌 Connecting to MetaTrader 5...")
        
        # Initialize MT5
        init_kwargs = {}
        if self.path:
            init_kwargs['path'] = self.path
        if self.login:
            init_kwargs['login'] = self.login
        if self.password:
            init_kwargs['password'] = self.password
        if self.server:
            init_kwargs['server'] = self.server
        
        if not mt5.initialize(**init_kwargs) if init_kwargs else mt5.initialize():
            self._last_error = f"MT5 initialization failed: {mt5.last_error()}"
            logger.error(self._last_error)
            return False
        
        # Verify connection
        account_info = mt5.account_info()
        if account_info is None:
            self._last_error = "Failed to get account info"
            logger.error(self._last_error)
            return False
        
        self._connected = True
        logger.info(f"✅ Connected to MT5")
        logger.info(f"   Account: {account_info.login}")
        logger.info(f"   Server: {account_info.server}")
        logger.info(f"   Balance: ${account_info.balance:,.2f}")
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from MT5."""
        if HAS_MT5 and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed")
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from MT5."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        # Normalize timeframe
        tf = timeframe.upper()
        if tf not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        mt5_tf = self.TIMEFRAME_MAP[tf]
        
        # Ensure symbol is available
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to select symbol {symbol}")
        
        # Fetch rates
        rates = mt5.copy_rates_range(symbol, mt5_tf, start, end)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol} {tf} from {start} to {end}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={'tick_volume': 'volume'})
        
        # Standardize
        df = self.validate_dataframe(df)
        
        logger.info(f"Fetched {len(df)} bars for {symbol} {tf}")
        return df
    
    def get_balance(self) -> float:
        """Get account balance in USD."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("Failed to get account info")
        
        return float(account_info.balance)
    
    def get_account_info(self) -> AccountInfo:
        """Get full account information."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Failed to get account info")
        
        return AccountInfo(
            balance=float(info.balance),
            equity=float(info.equity),
            margin_used=float(info.margin),
            margin_free=float(info.margin_free),
            currency=info.currency,
            leverage=info.leverage,
        )
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        volume: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> OrderResult:
        """Place an order on MT5."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Determine order type and price
        if order_type == OrderType.MARKET:
            mt5_order_type = mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            order_price = tick.ask if side == OrderSide.BUY else tick.bid
        else:
            # Limit orders
            if side == OrderSide.BUY:
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            order_price = price
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": order_price,
            "deviation": 20,
            "magic": 505050,
            "comment": "GoldenBreeze",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            raise RuntimeError(f"Order failed: {mt5.last_error()}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result.comment} (code: {result.retcode})")
        
        return OrderResult(
            order_id=str(result.order),
            symbol=symbol,
            side=side,
            volume=volume,
            price=result.price,
            status="filled",
            timestamp=datetime.now(timezone.utc),
            raw_response=result._asdict(),
        )
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            result.append(Position(
                symbol=pos.symbol,
                side=OrderSide.BUY if pos.type == mt5.POSITION_TYPE_BUY else OrderSide.SELL,
                volume=pos.volume,
                entry_price=pos.price_open,
                current_price=current_price,
                pnl=pos.profit,
                pnl_pct=(pos.profit / (pos.price_open * pos.volume)) * 100 if pos.price_open else 0,
                sl=pos.sl if pos.sl > 0 else None,
                tp=pos.tp if pos.tp > 0 else None,
            ))
        
        return result
    
    def close_position(
        self,
        symbol: str,
        volume: Optional[float] = None,
    ) -> OrderResult:
        """Close an open position."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        # Find position
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            raise ValueError(f"No position found for {symbol}")
        
        pos = positions[0]
        close_volume = volume if volume else pos.volume
        
        # Close direction is opposite
        if pos.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            close_price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 505050,
            "comment": "GoldenBreeze Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Close failed: {result.comment if result else mt5.last_error()}")
        
        return OrderResult(
            order_id=str(result.order),
            symbol=symbol,
            side=OrderSide.SELL if pos.type == mt5.POSITION_TYPE_BUY else OrderSide.BUY,
            volume=close_volume,
            price=result.price,
            status="filled",
            timestamp=datetime.now(timezone.utc),
            raw_response=result._asdict(),
        )
    
    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Get current bid/ask price."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Failed to get tick for {symbol}")
        
        return tick.bid, tick.ask

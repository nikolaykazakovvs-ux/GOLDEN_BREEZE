"""
🔌 Golden Breeze - Base Connector Abstract Class
================================================

Abstract base class for all exchange/platform connectors.
Provides unified interface for:
- MT5 (Forex/Gold)
- MEXC (Crypto Spot)
- TradeLocker (Prop Futures)

Author: Golden Breeze Team
Version: 1.0.0
Date: 2025-12-06
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class OrderResult:
    """Result of order placement."""
    order_id: str
    symbol: str
    side: OrderSide
    volume: float
    price: Optional[float]
    status: str
    timestamp: datetime
    raw_response: Optional[Dict] = None
    
    @property
    def is_filled(self) -> bool:
        return self.status.lower() in ('filled', 'executed', 'closed')


@dataclass
class Position:
    """Open position information."""
    symbol: str
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    
    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0


@dataclass
class AccountInfo:
    """Account information."""
    balance: float           # Total balance in USD
    equity: float            # Equity (balance + unrealized PnL)
    margin_used: float       # Used margin
    margin_free: float       # Free margin
    currency: str = "USD"
    leverage: int = 1


class BaseConnector(ABC):
    """
    Abstract base class for all exchange connectors.
    
    Provides unified interface for data fetching and order execution
    across different trading platforms.
    
    Usage:
        connector = MT5Connector()
        connector.connect()
        df = connector.get_history("XAUUSD", "M5", start, end)
        order = connector.place_order("XAUUSD", OrderSide.BUY, 0.01, sl=1900, tp=2000)
    """
    
    # Platform identification
    PLATFORM_NAME: str = "base"
    SUPPORTED_SYMBOLS: set = set()
    SUPPORTED_TIMEFRAMES: set = set()
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize connector.
        
        Args:
            credentials: Platform-specific credentials dict
        """
        self.credentials = credentials or {}
        self._connected = False
        self._last_error: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to platform."""
        return self._connected
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
    
    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the platform.
        
        Returns:
            True if connected successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the platform."""
        pass
    
    @abstractmethod
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD", "BTCUSDT")
            timeframe: Timeframe string (e.g., "M5", "H1", "1m", "1h")
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Sorted by timestamp ascending.
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> float:
        """
        Get account balance in USD.
        
        Returns:
            Balance as float (normalized to USD).
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get full account information.
        
        Returns:
            AccountInfo dataclass with balance, equity, margin, etc.
        """
        pass
    
    @abstractmethod
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
        """
        Place an order on the platform.
        
        Args:
            symbol: Trading symbol
            side: Buy or Sell
            volume: Order volume/size
            order_type: Market, Limit, Stop, etc.
            price: Limit/Stop price (required for limit orders)
            sl: Stop Loss price
            tp: Take Profit price
            
        Returns:
            OrderResult with order_id and status.
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> list[Position]:
        """
        Get all open positions.
        
        Returns:
            List of Position dataclasses.
        """
        pass
    
    @abstractmethod
    def close_position(
        self,
        symbol: str,
        volume: Optional[float] = None,
    ) -> OrderResult:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close
            volume: Volume to close (None = close all)
            
        Returns:
            OrderResult with close order status.
        """
        pass
    
    # =========================================================================
    # Helper Methods - Common implementations
    # =========================================================================
    
    def normalize_timeframe(self, tf: str) -> str:
        """
        Normalize timeframe string to platform-specific format.
        
        Override in subclass if platform uses different format.
        
        Args:
            tf: Input timeframe (e.g., "M5", "1m", "5m", "1h")
            
        Returns:
            Normalized timeframe string for this platform.
        """
        # Standard mapping: M5 <-> 5m, H1 <-> 1h, etc.
        tf_map = {
            # MT5 style -> CCXT style
            'M1': '1m', 'M5': '5m', 'M15': '15m', 'M30': '30m',
            'H1': '1h', 'H4': '4h', 'D1': '1d', 'W1': '1w',
            # CCXT style -> MT5 style
            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'W1',
        }
        return tf_map.get(tf, tf)
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to platform-specific format.
        
        Override in subclass for platform-specific normalization.
        
        Args:
            symbol: Input symbol (e.g., "XAUUSD", "BTC/USDT")
            
        Returns:
            Normalized symbol string for this platform.
        """
        return symbol.upper().replace('/', '')
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and standardize OHLCV DataFrame.
        
        Ensures consistent column names and data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardized DataFrame with columns:
            timestamp, open, high, low, close, volume
        """
        # Required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Column name mapping
        col_map = {
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vol': 'volume',
            'tick_volume': 'volume',
        }
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Check required columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Select and order columns
        df = df[required].copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
    
    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"<{self.__class__.__name__} [{self.PLATFORM_NAME}] {status}>"

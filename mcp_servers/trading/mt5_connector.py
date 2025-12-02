"""MT5 Connector — управление подключением к MetaTrader 5.

Обеспечивает инициализацию, авторизацию и безопасное отключение от MT5.
"""
from __future__ import annotations
import MetaTrader5 as mt5
from typing import Dict, Optional
from pathlib import Path
import json

class MT5Connector:
    """Singleton connector для MT5."""
    
    _instance: Optional['MT5Connector'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        pass
    
    def initialize(self, login: Optional[int] = None, password: Optional[str] = None, 
                   server: Optional[str] = None, path: Optional[str] = None) -> bool:
        """Инициализация MT5 и авторизация.
        
        Args:
            login: Номер счёта (если None, берётся из config)
            password: Пароль (если None, берётся из config)
            server: Сервер брокера (если None, берётся из config)
            path: Путь к terminal64.exe (опционально)
        
        Returns:
            True если успешно подключились
        """
        if self._initialized:
            return True
        
        # Если параметры не переданы, пытаемся загрузить из config
        if login is None or password is None or server is None:
            config = self._load_config()
            login = login or config.get("login")
            password = password or config.get("password")
            server = server or config.get("server")
        
        # Инициализация MT5
        if path:
            if not mt5.initialize(path=path):
                print(f"MT5 initialize() failed, error: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                print(f"MT5 initialize() failed, error: {mt5.last_error()}")
                return False
        
        # Авторизация если переданы учётные данные
        if login and password and server:
            if not mt5.login(login=login, password=password, server=server):
                print(f"MT5 login failed, error: {mt5.last_error()}")
                mt5.shutdown()
                return False
            print(f"Connected to MT5: {server}, account #{login}")
        else:
            print("MT5 initialized without login (existing connection)")
        
        self._initialized = True
        return True
    
    def shutdown(self):
        """Отключение от MT5."""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            print("MT5 connection closed")
    
    def is_connected(self) -> bool:
        """Проверка подключения."""
        if not self._initialized:
            return False
        # Попытка получить информацию об аккаунте
        account_info = mt5.account_info()
        return account_info is not None
    
    def get_account_info(self) -> Optional[Dict]:
        """Получить информацию об аккаунте."""
        if not self._initialized:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }
    
    def _load_config(self) -> Dict:
        """Загрузить конфигурацию MT5 из файла."""
        config_path = Path(__file__).parents[2] / "mt5_config.json"
        if not config_path.exists():
            return {}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to load MT5 config: {e}")
            return {}
    
    def __del__(self):
        """Автоматическое отключение при удалении объекта."""
        self.shutdown()


# Глобальный инстанс
_connector = MT5Connector()

def get_connector() -> MT5Connector:
    """Получить глобальный инстанс коннектора."""
    return _connector

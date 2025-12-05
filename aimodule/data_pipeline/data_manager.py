"""
Data Manager
Унифицированный менеджер данных для всех источников (MT5, MEXC, TradeLocker)
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union, Literal
import pandas as pd
import json

from aimodule.connector.base import BaseConnector
from aimodule.connector.mt5 import MT5Connector
from aimodule.connector.mexc import MEXCConnector
from aimodule.connector.tradelocker import TradeLockerConnector

logger = logging.getLogger(__name__)


# Типы источников
SourceType = Literal["mt5", "mexc", "tradelocker"]


class DataManager:
    """
    Унифицированный менеджер данных
    
    Позволяет:
    - Получать данные из любого источника через единый интерфейс
    - Сохранять данные в стандартизированном формате
    - Загружать сохранённые данные
    - Объединять данные из разных источников
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: Директория для сохранения данных.
                      По умолчанию: data/raw/{source}/{symbol}/
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш коннекторов
        self._connectors: dict[str, BaseConnector] = {}
        
    def get_connector(
        self,
        source: SourceType,
        **kwargs
    ) -> Optional[BaseConnector]:
        """
        Получение или создание коннектора
        
        Args:
            source: Тип источника ("mt5", "mexc", "tradelocker")
            **kwargs: Параметры для коннектора
            
        Returns:
            BaseConnector или None при ошибке
        """
        # Создаём ключ кэша
        cache_key = f"{source}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key in self._connectors:
            connector = self._connectors[cache_key]
            if connector.is_connected:
                return connector
        
        # Создаём новый коннектор
        try:
            if source == "mt5":
                connector = MT5Connector(**kwargs)
            elif source == "mexc":
                connector = MEXCConnector(**kwargs)
            elif source == "tradelocker":
                connector = TradeLockerConnector(**kwargs)
            else:
                logger.error(f"Неизвестный источник: {source}")
                return None
            
            # Подключаемся
            if connector.connect():
                self._connectors[cache_key] = connector
                return connector
            else:
                logger.error(f"Не удалось подключиться к {source}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания коннектора {source}: {e}")
            return None
    
    def fetch_data(
        self,
        source: SourceType,
        symbol: str,
        timeframe: str = "H1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 10000,
        save: bool = True,
        **connector_kwargs
    ) -> pd.DataFrame:
        """
        Получение данных из источника
        
        Args:
            source: Источник данных
            symbol: Торговый символ
            timeframe: Таймфрейм
            start_date: Начальная дата
            end_date: Конечная дата
            count: Количество баров (если start_date не указан)
            save: Сохранять ли данные в файл
            **connector_kwargs: Дополнительные параметры для коннектора
            
        Returns:
            DataFrame с OHLCV данными
        """
        connector = self.get_connector(source, **connector_kwargs)
        if not connector:
            return pd.DataFrame()
        
        # Получаем данные
        df = connector.get_history(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            count=count
        )
        
        if df.empty:
            logger.warning(f"Нет данных для {symbol} из {source}")
            return df
        
        # Добавляем метаданные
        df['source'] = source
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        # Сохраняем если нужно
        if save and not df.empty:
            self.save_data(df, source, symbol, timeframe)
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: str,
        append: bool = True
    ) -> str:
        """
        Сохранение данных в файл
        
        Args:
            df: DataFrame с данными
            source: Источник
            symbol: Символ
            timeframe: Таймфрейм
            append: Добавлять к существующим данным
            
        Returns:
            Путь к сохранённому файлу
        """
        # Создаём директорию
        save_dir = self.data_dir / source / self._normalize_filename(symbol)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Путь к файлу
        filename = f"{timeframe}.parquet"
        filepath = save_dir / filename
        
        # Если нужно добавить к существующим данным
        if append and filepath.exists():
            try:
                existing_df = pd.read_parquet(filepath)
                
                # Объединяем и убираем дубликаты
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['time'], keep='last')
                df = df.sort_values('time').reset_index(drop=True)
                
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующие данные: {e}")
        
        # Сохраняем
        df.to_parquet(filepath, index=False)
        
        logger.info(f"💾 Сохранено {len(df)} баров в {filepath}")
        
        # Также сохраняем метаданные
        self._save_metadata(save_dir, source, symbol, timeframe, len(df))
        
        return str(filepath)
    
    def load_data(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Загрузка сохранённых данных
        
        Args:
            source: Источник
            symbol: Символ
            timeframe: Таймфрейм
            start_date: Фильтр по начальной дате
            end_date: Фильтр по конечной дате
            
        Returns:
            DataFrame с данными
        """
        filepath = self.data_dir / source / self._normalize_filename(symbol) / f"{timeframe}.parquet"
        
        if not filepath.exists():
            logger.warning(f"Файл не найден: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(filepath)
            
            # Применяем фильтры по дате
            if start_date:
                if df['time'].dt.tz is None:
                    start_date = start_date.replace(tzinfo=None)
                df = df[df['time'] >= start_date]
            
            if end_date:
                if df['time'].dt.tz is None:
                    end_date = end_date.replace(tzinfo=None)
                df = df[df['time'] <= end_date]
            
            logger.info(f"📂 Загружено {len(df)} баров из {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame()
    
    def list_available_data(self) -> dict:
        """
        Список доступных сохранённых данных
        
        Returns:
            Словарь {source: {symbol: [timeframes]}}
        """
        result = {}
        
        for source_dir in self.data_dir.iterdir():
            if not source_dir.is_dir():
                continue
            
            source = source_dir.name
            result[source] = {}
            
            for symbol_dir in source_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                
                symbol = symbol_dir.name
                timeframes = []
                
                for file in symbol_dir.glob("*.parquet"):
                    timeframes.append(file.stem)
                
                if timeframes:
                    result[source][symbol] = sorted(timeframes)
        
        return result
    
    def fetch_training_data(
        self,
        source: SourceType,
        symbol: str,
        timeframes: list[str] = ["M15", "H1", "H4"],
        days_back: int = 365,
        **connector_kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Получение данных для обучения модели
        
        Args:
            source: Источник данных
            symbol: Торговый символ
            timeframes: Список таймфреймов
            days_back: Сколько дней истории загружать
            **connector_kwargs: Параметры коннектора
            
        Returns:
            Словарь {timeframe: DataFrame}
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        result = {}
        
        for tf in timeframes:
            logger.info(f"📊 Загрузка {symbol} {tf} за {days_back} дней...")
            
            df = self.fetch_data(
                source=source,
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                save=True,
                **connector_kwargs
            )
            
            if not df.empty:
                result[tf] = df
                logger.info(f"   → {len(df)} баров")
            else:
                logger.warning(f"   → Нет данных")
        
        return result
    
    def merge_multi_timeframe(
        self,
        dataframes: dict[str, pd.DataFrame],
        base_timeframe: str = "M15"
    ) -> pd.DataFrame:
        """
        Объединение данных разных таймфреймов в один DataFrame
        
        Args:
            dataframes: Словарь {timeframe: DataFrame}
            base_timeframe: Базовый таймфрейм (самый мелкий)
            
        Returns:
            Объединённый DataFrame
        """
        if base_timeframe not in dataframes:
            logger.error(f"Базовый таймфрейм {base_timeframe} не найден")
            return pd.DataFrame()
        
        # Берём базовый DF
        result = dataframes[base_timeframe].copy()
        
        # Переименовываем колонки
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        rename_map = {col: f"{col}_{base_timeframe}" for col in base_cols}
        result = result.rename(columns=rename_map)
        
        # Добавляем данные других таймфреймов
        for tf, df in dataframes.items():
            if tf == base_timeframe:
                continue
            
            # Ресемплим к базовому таймфрейму (forward fill)
            df = df.copy()
            df = df.set_index('time')
            
            # Добавляем колонки с префиксом таймфрейма
            for col in base_cols:
                if col in df.columns:
                    col_name = f"{col}_{tf}"
                    
                    # Для каждой строки базового DF находим последнее значение из старшего TF
                    result[col_name] = result['time'].apply(
                        lambda t: self._get_last_value(df, col, t)
                    )
        
        return result
    
    def _get_last_value(self, df: pd.DataFrame, column: str, timestamp: datetime) -> float:
        """Получение последнего значения колонки до указанного времени"""
        try:
            # Находим все значения до timestamp
            mask = df.index <= timestamp
            if mask.any():
                return df.loc[mask, column].iloc[-1]
            return 0.0
        except:
            return 0.0
    
    def _normalize_filename(self, symbol: str) -> str:
        """Нормализация символа для имени файла"""
        return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    def _save_metadata(
        self,
        save_dir: Path,
        source: str,
        symbol: str,
        timeframe: str,
        count: int
    ):
        """Сохранение метаданных"""
        meta_file = save_dir / "metadata.json"
        
        metadata = {
            "source": source,
            "symbol": symbol,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "timeframes": {}
        }
        
        # Загружаем существующие метаданные
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Обновляем информацию о таймфрейме
        metadata["timeframes"][timeframe] = {
            "count": count,
            "updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Сохраняем
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def disconnect_all(self):
        """Отключение всех коннекторов"""
        for connector in self._connectors.values():
            try:
                connector.disconnect()
            except:
                pass
        self._connectors.clear()
        logger.info("Все коннекторы отключены")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_all()


# Удобные функции для быстрого использования

def fetch_mt5_data(
    symbol: str,
    timeframe: str = "H1",
    days_back: int = 30,
    **kwargs
) -> pd.DataFrame:
    """Быстрое получение данных из MT5"""
    dm = DataManager()
    return dm.fetch_data(
        source="mt5",
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime.now(timezone.utc) - timedelta(days=days_back),
        **kwargs
    )


def fetch_crypto_data(
    symbol: str,
    timeframe: str = "1h",
    days_back: int = 30,
    **kwargs
) -> pd.DataFrame:
    """Быстрое получение криптовалютных данных"""
    dm = DataManager()
    return dm.fetch_data(
        source="mexc",
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime.now(timezone.utc) - timedelta(days=days_back),
        **kwargs
    )

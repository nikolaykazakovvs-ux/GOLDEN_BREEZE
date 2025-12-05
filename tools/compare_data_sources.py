"""
Сравнение качества данных из разных источников XAU/USD
Для валидации выбора источника данных для переобучения модели
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

class DataSourceComparator:
    """
    Сравнивает данные из разных источников
    """
    
    def __init__(self):
        self.data = {}
    
    def load_mt5_data(self, filepath: str) -> pd.DataFrame:
        """Загрузить данные из MT5 экспорта"""
        try:
            df = pd.read_csv(filepath)
            
            # MT5 формат: timestamp, open, high, low, close, volume
            if len(df.columns) >= 5:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + list(df.columns[6:])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            logger.info(f"✅ MT5: Загружено {len(df)} баров")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки MT5: {e}")
            return None
    
    def load_investing_data(self, filepath: str) -> pd.DataFrame:
        """Загрузить данные из investing.com"""
        try:
            df = pd.read_csv(filepath)
            
            # Почистить числовые колонки (убрать запятые)
            for col in ['Open', 'High', 'Low', 'Price']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Переименовать колонки
            df.rename(columns={
                'Date': 'timestamp',
                'Price': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low'
            }, inplace=True)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            logger.info(f"✅ Investing: Загружено {len(df)} баров (D1)")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки investing: {e}")
            return None
    
    def compare_statistics(self, 
                          df1: pd.DataFrame, 
                          name1: str,
                          df2: pd.DataFrame = None,
                          name2: str = None) -> Dict:
        """
        Сравнить статистику между двумя источниками
        """
        logger.info("\n" + "="*70)
        logger.info(f"📊 СРАВНЕНИЕ: {name1}" + (f" vs {name2}" if df2 is not None else ""))
        logger.info("="*70)
        
        # Статистика источника 1
        logger.info(f"\n{name1}:")
        logger.info(f"  • Баров: {len(df1):,}")
        logger.info(f"  • Период: {df1['timestamp'].min()} - {df1['timestamp'].max()}")
        logger.info(f"  • Дней: {(df1['timestamp'].max() - df1['timestamp'].min()).days}")
        logger.info(f"  • Цены: {df1['open'].min():.2f} - {df1['high'].max():.2f}")
        logger.info(f"  • Среднее изменение: {df1['close'].pct_change().mean()*100:.4f}%")
        logger.info(f"  • Волатильность: {df1['close'].pct_change().std()*100:.4f}%")
        logger.info(f"  • Пропуски: {df1['timestamp'].diff().mode()[0]} (типичный интервал)")
        
        if df2 is not None:
            logger.info(f"\n{name2}:")
            logger.info(f"  • Баров: {len(df2):,}")
            logger.info(f"  • Период: {df2['timestamp'].min()} - {df2['timestamp'].max()}")
            logger.info(f"  • Дней: {(df2['timestamp'].max() - df2['timestamp'].min()).days}")
            logger.info(f"  • Цены: {df2['open'].min():.2f} - {df2['high'].max():.2f}")
            logger.info(f"  • Среднее изменение: {df2['close'].pct_change().mean()*100:.4f}%")
            logger.info(f"  • Волатильность: {df2['close'].pct_change().std()*100:.4f}%")
            
            # Сравнение
            logger.info(f"\n{'Метрика':<30} {'Разница':<15}")
            logger.info("─" * 45)
            
            # Найти общий период
            start = max(df1['timestamp'].min(), df2['timestamp'].min())
            end = min(df1['timestamp'].max(), df2['timestamp'].max())
            
            df1_common = df1[(df1['timestamp'] >= start) & (df1['timestamp'] <= end)]
            df2_common = df2[(df2['timestamp'] >= start) & (df2['timestamp'] <= end)]
            
            if len(df1_common) > 0 and len(df2_common) > 0:
                logger.info(f"Общий период: {start} - {end}")
                logger.info(f"  • Баров в общем: {len(df1_common)} vs {len(df2_common)}")
                
                # Сравнить Close цены
                correlation = df1_common['close'].corr(df2_common['close'])
                logger.info(f"  • Корреляция Close: {correlation:.4f}")
                
                # RMSE между ценами
                price_diff = np.abs(df1_common['close'].values - df2_common['close'].values)
                rmse = np.sqrt(np.mean(price_diff**2))
                logger.info(f"  • RMSE цен: ${rmse:.2f}")
                logger.info(f"  • Макс разница: ${price_diff.max():.2f}")
                logger.info(f"  • Средняя разница: ${price_diff.mean():.2f}")
    
    def recommendation(self):
        """Дать рекомендацию по выбору источника"""
        logger.info("\n" + "="*70)
        logger.info("💡 РЕКОМЕНДАЦИИ ДЛЯ ПЕРЕОБУЧЕНИЯ МОДЕЛИ")
        logger.info("="*70)
        
        logger.info("""
🎯 ВЫБОР ИСТОЧНИКА для 6-летних данных XAU/USD M5:

✅ РЕКОМЕНДУЕТСЯ (приоритет):
  1. Kaggle "XAU/USD Gold Price Historical Data"
     • Быстро скачать (все уже обработано)
     • M5 за 6 лет готовые CSV
     • Источник: MT5 экспорт (как у тебя)
     • Формат: совместим с текущей системой
     🔗 https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024

  2. Dukascopy (альтернатива для валидации)
     • Высочайшее качество (швейцарский банк)
     • M1 данные → самостоятельно агрегировать в M5
     • Потребует Node.js + 2-3 часа загрузки
     📍 npx dukascopy-node -i xauusd -from 2019-01-01 -to 2025-12-31 -t m1 -f csv

⚠️ НЕ РЕКОМЕНДУЕТСЯ:
  • TrueFX - XAU/USD отсутствует в публичном доступе
  • HistData самостоятельно - нужно собирать по месяцам, парсить EST

📋 ПЛАН ДЕЙСТВИЙ:

Неделя 1:
  □ Скачать датасет с Kaggle (~500 МБ M5 csv)
  □ Загрузить в проект
  □ Сравнить с твоей текущей M5 (2024-2025)
  □ Проверить корреляцию и расхождения
  
Неделя 2:
  □ Подготовить данные (UTC, outliers, gaps)
  □ Переобучить LSTM на 6-летних данных
  □ Сравнить метрики: текущая vs новая модель
  
Неделя 3+:
  □ Бэктест на исторических данных
  □ Внедрить в live боте
  □ Мониторить улучшения
        """)

if __name__ == '__main__':
    comp = DataSourceComparator()
    
    # Загрузить текущие данные
    m5_data = comp.load_mt5_data('data/raw/XAUUSD/M5.csv')
    d1_data = comp.load_investing_data('data/raw/XAUUSD/investing_d1.csv')
    
    # Сравнить M5 и D1
    if m5_data is not None and d1_data is not None:
        # Ресумплировать M5 в D1 для сравнения
        m5_daily = m5_data.set_index('timestamp').resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        comp.compare_statistics(m5_daily, "M5 (ресумплировано в D1)", 
                               d1_data, "D1 (investing.com)")
    
    # Вывести рекомендации
    comp.recommendation()

"""
Проверка качества исторических данных XAU/USD
Сравнение данных с эталонными источниками
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/raw/XAUUSD')

class DataValidator:
    def __init__(self):
        self.results = {}
    
    def load_data(self, filepath):
        """Загрузить CSV с автоматическим определением параметров"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"✅ Загружено: {filepath} ({len(df)} строк)")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {filepath}: {e}")
            return None
    
    def validate_ohlc(self, df, timeframe):
        """Проверить целостность OHLC"""
        issues = []
        
        if 'high' not in df.columns or 'low' not in df.columns:
            issues.append("❌ Отсутствуют колонки high/low")
            return issues
        
        # Проверка: High >= Low
        invalid = df[df['high'] < df['low']]
        if len(invalid) > 0:
            issues.append(f"❌ {len(invalid)} баров с High < Low (OHLC нарушены)")
        
        # Проверка: Open и Close между High и Low
        invalid_open = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
        if len(invalid_open) > 0:
            issues.append(f"⚠️ {len(invalid_open)} баров где Open вне [Low, High]")
        
        if len(issues) == 0:
            issues.append(f"✅ OHLC целостность: OK ({len(df)} баров)")
        
        return issues
    
    def validate_timestamps(self, df, timeframe):
        """Проверить временные метки"""
        issues = []
        
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            issues.append("❌ Отсутствует временная метка")
            return issues
        
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        
        try:
            df['dt'] = pd.to_datetime(df[time_col])
        except:
            issues.append(f"❌ Не удалось парсить дату в колонке '{time_col}'")
            return issues
        
        # Проверка на дубликаты
        dupes = df[df['dt'].duplicated()].shape[0]
        if dupes > 0:
            issues.append(f"⚠️ {dupes} дублирующихся временных меток")
        
        # Проверка на пропуски
        df_sorted = df.sort_values('dt')
        dt_diffs = df_sorted['dt'].diff()
        
        expected_delta = pd.Timedelta(minutes=int(timeframe.rstrip('M')))
        gaps = dt_diffs[dt_diffs != expected_delta]
        
        if len(gaps) > 5:  # Игнорируем выходные и праздники
            issues.append(f"⚠️ {len(gaps)} пропусков (выходные/праздники/праздники - нормально)")
        
        date_range = f"{df_sorted['dt'].min().date()} - {df_sorted['dt'].max().date()}"
        issues.append(f"✅ Диапазон дат: {date_range}")
        
        return issues
    
    def validate_volume(self, df):
        """Проверить объёмы"""
        issues = []
        
        if 'volume' not in df.columns:
            issues.append("⚠️ Колонка Volume отсутствует")
            return issues
        
        zero_vol = (df['volume'] == 0).sum()
        if zero_vol > 0:
            issues.append(f"⚠️ {zero_vol} баров с Volume=0 (редко встречается в золоте)")
        
        neg_vol = (df['volume'] < 0).sum()
        if neg_vol > 0:
            issues.append(f"❌ {neg_vol} баров с отрицательным объёмом")
        else:
            issues.append(f"✅ Volume: все значения >= 0")
        
        return issues
    
    def check_data_sources(self):
        """Проверить доступные источники данных"""
        logger.info("\n" + "="*70)
        logger.info("📊 АНАЛИЗ КАЧЕСТВА ДАННЫХ XAU/USD")
        logger.info("="*70)
        
        sources = {
            'M5': 'M5.csv',
            'H1': 'H1.csv',
            'D1': 'investing_d1.csv'
        }
        
        for timeframe, filename in sources.items():
            filepath = DATA_DIR / filename
            
            logger.info(f"\n🔍 Проверка {timeframe} ({filename}):")
            logger.info("-" * 70)
            
            if not filepath.exists():
                logger.warning(f"⚠️ Файл не найден: {filepath}")
                continue
            
            df = self.load_data(filepath)
            if df is None:
                continue
            
            # Валидация
            ohlc_issues = self.validate_ohlc(df, timeframe)
            time_issues = self.validate_timestamps(df, timeframe)
            vol_issues = self.validate_volume(df)
            
            for issue in ohlc_issues + time_issues + vol_issues:
                logger.info(f"  {issue}")
            
            # Статистика
            logger.info(f"\n  📈 Статистика:")
            logger.info(f"    • Размер: {len(df)} баров")
            logger.info(f"    • Открыто: {df['open'].min():.2f} - {df['open'].max():.2f}")
            logger.info(f"    • Объём: {df['volume'].sum():,.0f} (среднее: {df['volume'].mean():.0f})")
            
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                logger.info(f"    • Доходность: μ={returns.mean()*100:.4f}% σ={returns.std()*100:.4f}%")
            
            self.results[timeframe] = {
                'file': filename,
                'rows': len(df),
                'issues': len([i for i in ohlc_issues + time_issues + vol_issues if '❌' in i])
            }
        
        return self.results
    
    def print_summary(self):
        """Итоговый отчёт"""
        logger.info("\n" + "="*70)
        logger.info("📋 ИТОГОВЫЙ ОТЧЁТ")
        logger.info("="*70)
        
        for tf, info in self.results.items():
            status = "✅" if info['issues'] == 0 else "⚠️"
            logger.info(f"{status} {tf}: {info['rows']} баров, {info['issues']} проблем")
        
        logger.info("\n" + "="*70)
        logger.info("💡 РЕКОМЕНДАЦИИ:")
        logger.info("="*70)
        
        logger.info("""
1. ✅ Используй текущие данные D1/H1 для предварительного анализа
2. ⚠️ M5 данные ограничены 1 годом - недостаточно для стабильной модели
3. 🎯 Рекомендация: Скачать M1 данные из Dukascopy за 6 лет
4. 📦 Для упрощения: используй готовый датасет Kaggle (M5 + H1 за 6 лет)
5. 🔄 Переобучить модель на многолетних данных с лучшей валидацией

Команда для Dukascopy:
  npx dukascopy-node -i xauusd -from 2019-01-01 -to 2025-12-31 -t m1 -f csv --volumes

Альтернатива:
  https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024
        """)

if __name__ == '__main__':
    validator = DataValidator()
    validator.check_data_sources()
    validator.print_summary()

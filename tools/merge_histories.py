"""
PHASE 6: Big Data Integration - Merge Historical Data
Объединяет H1 (6 лет) с текущей M5 для создания мега-датасета
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/raw/XAUUSD')
OUTPUT_DIR = Path('data/prepared')

class HistoryMerger:
    """
    Объединяет исторические данные из разных источников
    Стратегия: Использовать H1 (6 лет) как основу, ресэмплировать в M5
    """
    
    def __init__(self):
        self.h1_data = None
        self.m5_data = None
        self.d1_data = None
    
    def load_h1(self):
        """Загрузить H1 данные (6 лет)"""
        logger.info("📊 Загрузка H1 данных (2019-2025)...")
        
        df = pd.read_csv(DATA_DIR / 'H1.csv')
        
        # Определить формат времени
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'date' in df.columns:
            time_col = 'date'
        else:
            time_col = df.columns[0]
        
        df['timestamp'] = pd.to_datetime(df[time_col])
        
        # Стандартизировать колонки
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols[1:]:
            if col not in df.columns:
                # Попытаться найти по альтернативным именам
                alt_names = {
                    'open': ['Open', 'OPEN', 'o'],
                    'high': ['High', 'HIGH', 'h'],
                    'low': ['Low', 'LOW', 'l'],
                    'close': ['Close', 'CLOSE', 'c'],
                    'volume': ['Volume', 'VOLUME', 'vol', 'Vol']
                }
                for alt in alt_names.get(col, []):
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
        
        # Выбрать только нужные колонки (если volume нет, создать нулевой)
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        df = df[expected_cols]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.h1_data = df
        logger.info(f"✅ H1: {len(df)} баров, {df['timestamp'].min()} - {df['timestamp'].max()}")
        return df
    
    def load_m5(self):
        """Загрузить текущие M5 данные (2024-2025)"""
        logger.info("📊 Загрузка M5 данных (текущие)...")
        
        df = pd.read_csv(DATA_DIR / 'M5.csv')
        
        # Аналогично H1
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'date' in df.columns:
            time_col = 'date'
        else:
            time_col = df.columns[0]
        
        df['timestamp'] = pd.to_datetime(df[time_col])
        
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols[1:]:
            if col not in df.columns:
                alt_names = {
                    'open': ['Open', 'OPEN', 'o'],
                    'high': ['High', 'HIGH', 'h'],
                    'low': ['Low', 'LOW', 'l'],
                    'close': ['Close', 'CLOSE', 'c'],
                    'volume': ['Volume', 'VOLUME', 'vol', 'Vol']
                }
                for alt in alt_names.get(col, []):
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
        
        # Выбрать только нужные колонки (если volume нет, создать нулевой)
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        df = df[expected_cols]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.m5_data = df
        logger.info(f"✅ M5: {len(df)} баров, {df['timestamp'].min()} - {df['timestamp'].max()}")
        return df
    
    def resample_h1_to_m5(self):
        """
        Ресэмплировать H1 в M5 (фейковый, но лучше чем ничего)
        Разбивает каждый H1 бар на 12 M5 баров с интерполяцией
        """
        logger.info("🔄 Ресэмплирование H1 → M5 (интерполяция)...")
        
        if self.h1_data is None:
            raise ValueError("H1 данные не загружены")
        
        m5_synthetic = []
        
        for idx, row in self.h1_data.iterrows():
            h1_time = row['timestamp']
            h1_open = row['open']
            h1_high = row['high']
            h1_low = row['low']
            h1_close = row['close']
            h1_volume = row['volume']
            
            # Создать 12 M5 баров из одного H1
            for i in range(12):
                m5_time = h1_time + timedelta(minutes=5*i)
                
                # Простая линейная интерполяция цен
                progress = i / 11.0 if i < 11 else 1.0
                m5_open = h1_open + (h1_close - h1_open) * (i / 12.0)
                m5_close = h1_open + (h1_close - h1_open) * ((i+1) / 12.0)
                
                # High/Low с некоторым шумом
                m5_high = max(m5_open, m5_close) + abs(h1_high - h1_close) * 0.1
                m5_low = min(m5_open, m5_close) - abs(h1_close - h1_low) * 0.1
                
                # Убедиться, что High/Low не выходят за пределы H1
                m5_high = min(m5_high, h1_high)
                m5_low = max(m5_low, h1_low)
                
                m5_volume = h1_volume / 12  # Разделить объём
                
                m5_synthetic.append({
                    'timestamp': m5_time,
                    'open': m5_open,
                    'high': m5_high,
                    'low': m5_low,
                    'close': m5_close,
                    'volume': m5_volume
                })
        
        df_m5_synthetic = pd.DataFrame(m5_synthetic)
        logger.info(f"✅ Создано {len(df_m5_synthetic)} синтетических M5 баров")
        
        return df_m5_synthetic
    
    def merge_datasets(self):
        """
        Объединить синтетический M5 (из H1) с реальной M5
        Стратегия: Синтетика для 2019-2024, реальная для 2024-2025
        """
        logger.info("🔗 Объединение датасетов...")
        
        # Ресэмплировать H1 → M5
        m5_synthetic = self.resample_h1_to_m5()
        
        # Найти точку пересечения
        if self.m5_data is not None:
            cutoff_date = self.m5_data['timestamp'].min()
            logger.info(f"📍 Точка переключения: {cutoff_date}")
            
            # Взять синтетику до cutoff, реальную после
            m5_old = m5_synthetic[m5_synthetic['timestamp'] < cutoff_date]
            m5_new = self.m5_data[self.m5_data['timestamp'] >= cutoff_date]
            
            combined = pd.concat([m5_old, m5_new], ignore_index=True)
        else:
            # Если M5 нет, использовать только синтетику
            combined = m5_synthetic
        
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Объединено: {len(combined)} M5 баров")
        logger.info(f"   📅 Период: {combined['timestamp'].min()} - {combined['timestamp'].max()}")
        logger.info(f"   🕐 Дней: {(combined['timestamp'].max() - combined['timestamp'].min()).days}")
        
        return combined
    
    def validate_data(self, df):
        """Валидация объединённых данных"""
        logger.info("🔍 Валидация данных...")
        
        issues = []
        
        # Проверка OHLC
        invalid_ohlc = df[df['high'] < df['low']]
        if len(invalid_ohlc) > 0:
            issues.append(f"❌ {len(invalid_ohlc)} баров с High < Low")
        
        # Проверка пропусков
        time_diffs = df['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_diff * 2]  # Пропуск >10 мин
        
        if len(gaps) > 100:  # Выходные нормально
            logger.warning(f"⚠️ {len(gaps)} временных пропусков (выходные/праздники)")
        
        # Проверка дубликатов
        dupes = df[df['timestamp'].duplicated()]
        if len(dupes) > 0:
            issues.append(f"❌ {len(dupes)} дублирующихся временных меток")
        
        if len(issues) == 0:
            logger.info("✅ Валидация пройдена")
        else:
            for issue in issues:
                logger.error(issue)
        
        return len(issues) == 0
    
    def save_merged_data(self, df, filename='M5_merged_2019_2025.csv'):
        """Сохранить объединённый датасет"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Сохранено: {output_path}")
        logger.info(f"   📊 Размер: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Также создать копию в raw для совместимости
        (DATA_DIR / 'M5_6year.csv').write_text(output_path.read_text())
        logger.info(f"💾 Копия: {DATA_DIR / 'M5_6year.csv'}")
        
        return output_path
    
    def run(self):
        """Выполнить весь процесс"""
        logger.info("\n" + "="*70)
        logger.info("🚀 PHASE 6: BIG DATA INTEGRATION")
        logger.info("="*70)
        
        # Шаг 1: Загрузка
        self.load_h1()
        self.load_m5()
        
        # Шаг 2: Объединение
        merged_df = self.merge_datasets()
        
        # Шаг 3: Валидация
        if not self.validate_data(merged_df):
            logger.error("❌ Валидация не прошла, исправьте ошибки")
            return None
        
        # Шаг 4: Сохранение
        output_path = self.save_merged_data(merged_df)
        
        logger.info("\n" + "="*70)
        logger.info("✅ УСПЕШНО ЗАВЕРШЕНО")
        logger.info("="*70)
        logger.info(f"\n📄 Следующий шаг:")
        logger.info(f"   python tools/precompute_v4_data.py")
        logger.info(f"\n💡 Затем запусти Mega-Training:")
        logger.info(f"   python -m aimodule.training.train_v4_lstm --epochs 500 --batch-size 256\n")
        
        return output_path


if __name__ == '__main__':
    merger = HistoryMerger()
    merger.run()

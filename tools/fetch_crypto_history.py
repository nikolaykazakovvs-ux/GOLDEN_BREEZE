"""
Fetch Crypto History from Binance Exchange
===========================================

Downloads historical OHLCV data for BTC/USDT (and other pairs)
from Binance exchange using pagination to get full 4-year history.

Note: MEXC only stores ~500 days of history. Binance has full history since 2017.

Usage:
    python tools/fetch_crypto_history.py

Output:
    data/raw/BINANCE/BTC_USDT/M5.csv
    data/raw/BINANCE/BTC_USDT/H1.csv
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import logging

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

try:
    import ccxt
except ImportError:
    print("❌ ccxt не установлен. Запустите: pip install ccxt")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

EXCHANGE = "binance"  # binance имеет полную историю с 2017
SYMBOL = "BTC/USDT"
SYMBOL_DIR = "BTC_USDT"  # Для имени директории (без слеша)

# Какие таймфреймы качать
TIMEFRAMES = ["5m", "1h"]  # ccxt формат

# Даты (4 года назад)
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)

# Лимиты Binance API
MAX_CANDLES_PER_REQUEST = 1000  # Максимум свечей за один запрос

# Задержка между запросами (чтобы не превысить rate limit)
REQUEST_DELAY = 0.2  # секунд


# ============================================================================
# TIMEFRAME UTILITIES
# ============================================================================

def get_timeframe_minutes(tf: str) -> int:
    """Возвращает количество минут в таймфрейме (ccxt формат)"""
    tf_minutes = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }
    return tf_minutes.get(tf, 5)


def calculate_batch_end(start: datetime, tf: str, count: int) -> datetime:
    """Вычисляет конец батча для пагинации"""
    minutes = get_timeframe_minutes(tf)
    return start + timedelta(minutes=minutes * count)


# ============================================================================
# MAIN FETCH FUNCTION
# ============================================================================

def fetch_full_history(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Загружает полную историю с пагинацией.
    
    Args:
        exchange: Подключённый ccxt exchange
        symbol: Торговая пара (например "BTC/USDT")
        timeframe: Таймфрейм (5m, 1h, etc.)
        start_date: Начальная дата
        end_date: Конечная дата
        
    Returns:
        DataFrame с полной историей
    """
    logger.info(f"📥 Загрузка {symbol} {timeframe}")
    logger.info(f"   Период: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    all_data = []
    current_since = int(start_date.timestamp() * 1000)  # миллисекунды
    end_ms = int(end_date.timestamp() * 1000)
    batch_num = 0
    
    # Минут в одном баре
    bar_minutes = get_timeframe_minutes(timeframe)
    bar_ms = bar_minutes * 60 * 1000  # в миллисекундах
    
    while current_since < end_ms:
        batch_num += 1
        
        try:
            # Получаем batch
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=MAX_CANDLES_PER_REQUEST
            )
        except Exception as e:
            logger.error(f"   Ошибка запроса: {e}")
            time.sleep(1)
            continue
        
        if not ohlcv:
            logger.warning(f"   Batch {batch_num}: пустой ответ, завершаем")
            break
        
        # Конвертируем в DataFrame
        df_batch = pd.DataFrame(
            ohlcv,
            columns=['time', 'open', 'high', 'low', 'close', 'volume']
        )
        df_batch['time'] = pd.to_datetime(df_batch['time'], unit='ms', utc=True)
        
        all_data.append(df_batch)
        
        # Находим последнюю дату в batch
        last_time_ms = ohlcv[-1][0]
        current_since = last_time_ms + bar_ms  # следующий бар
        
        # Прогресс
        progress = min(100, (current_since - int(start_date.timestamp() * 1000)) / 
                      (end_ms - int(start_date.timestamp() * 1000)) * 100)
        logger.info(f"   Batch {batch_num}: +{len(ohlcv)} bars | Total: {sum(len(d) for d in all_data):,} | Progress: {progress:.1f}%")
        
        # Rate limit
        time.sleep(REQUEST_DELAY)
        
        # Safety: если получили меньше чем лимит, значит дошли до конца
        if len(ohlcv) < MAX_CANDLES_PER_REQUEST:
            logger.info(f"   Достигнут конец данных")
            break
    
    if not all_data:
        logger.error(f"❌ Нет данных для {symbol} {timeframe}")
        return pd.DataFrame()
    
    # Объединяем все батчи
    df_full = pd.concat(all_data, ignore_index=True)
    
    # Удаляем дубликаты по времени
    df_full = df_full.drop_duplicates(subset=['time'], keep='first')
    
    # Сортируем
    df_full = df_full.sort_values('time').reset_index(drop=True)
    
    # Фильтруем по end_date
    df_full = df_full[df_full['time'] <= end_date]
    
    # Добавляем tick_volume как копию volume (для совместимости с нашими моделями)
    df_full['tick_volume'] = df_full['volume']
    
    logger.info(f"✅ {symbol} {timeframe}: всего {len(df_full):,} баров")
    
    return df_full


def save_to_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Сохраняет DataFrame в CSV"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Конвертируем время в строку для CSV
    df_save = df.copy()
    df_save['time'] = df_save['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_save.to_csv(filepath, index=False)
    
    # Размер файла
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"💾 Сохранено: {filepath} ({size_mb:.2f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BINANCE CRYPTO HISTORY DOWNLOADER")
    print("=" * 70)
    print(f"Exchange: {EXCHANGE.upper()}")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    # 1. Подключаемся к Binance (публичный доступ, без ключей)
    logger.info(f"🔌 Подключение к {EXCHANGE.upper()}...")
    
    exchange = getattr(ccxt, EXCHANGE)({
        'enableRateLimit': True,
        'rateLimit': 100,
    })
    
    # Загружаем рынки
    exchange.load_markets()
    logger.info(f"✅ Подключено к {EXCHANGE.upper()}. Доступно {len(exchange.markets)} рынков")
    
    # Проверяем текущую цену
    ticker = exchange.fetch_ticker(SYMBOL)
    logger.info(f"📊 {SYMBOL} текущая цена: ${ticker['last']:,.2f}")
    
    # 2. Качаем каждый таймфрейм
    output_dir = PROJECT_ROOT / "data" / "raw" / EXCHANGE.upper() / SYMBOL_DIR
    
    for tf in TIMEFRAMES:
        print()
        logger.info(f"{'='*50}")
        logger.info(f"FETCHING {tf.upper()}")
        logger.info(f"{'='*50}")
        
        df = fetch_full_history(
            exchange=exchange,
            symbol=SYMBOL,
            timeframe=tf,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        if df.empty:
            logger.error(f"❌ Не удалось загрузить {tf}")
            continue
        
        # Сохраняем с именем в формате M5, H1 для совместимости
        tf_name = tf.upper().replace("M", "M").replace("H", "H")
        if tf == "5m":
            tf_name = "M5"
        elif tf == "1h":
            tf_name = "H1"
        elif tf == "4h":
            tf_name = "H4"
        elif tf == "1d":
            tf_name = "D1"
            
        csv_path = output_dir / f"{tf_name}.csv"
        save_to_csv(df, csv_path)
        
        # Статистика
        print(f"\n📈 Статистика {tf_name}:")
        print(f"   • Первая свеча: {df['time'].min()}")
        print(f"   • Последняя свеча: {df['time'].max()}")
        print(f"   • Всего баров: {len(df):,}")
        print(f"   • Open range: ${df['open'].min():,.2f} - ${df['open'].max():,.2f}")
    
    print()
    print("=" * 70)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nДанные сохранены в: {output_dir}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

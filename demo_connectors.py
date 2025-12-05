"""
Demo: Unified Multi-Connector System
Демонстрация работы унифицированной системы коннекторов

Показывает:
1. Подключение к разным источникам (MT5, MEXC)
2. Получение данных через единый интерфейс
3. Сохранение и загрузка данных
"""

import logging
from datetime import datetime, timezone, timedelta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from aimodule.connector import (
    MT5Connector,
    MEXCConnector,
    OrderSide,
    OrderType
)
from aimodule.data_pipeline.data_manager import DataManager

def demo_mt5():
    """Демо работы с MT5"""
    print("\n" + "="*60)
    print("📊 ДЕМО: MT5 Connector")
    print("="*60)
    
    connector = MT5Connector()
    
    if not connector.connect():
        print("❌ MT5 не установлен или недоступен")
        return None
    
    print("✅ MT5 подключен")
    
    # Получаем информацию об аккаунте
    account = connector.get_account_info()
    print(f"\n📊 Аккаунт:")
    print(f"   Баланс: ${account.balance:,.2f}")
    print(f"   Эквити: ${account.equity:,.2f}")
    print(f"   Маржа: ${account.margin:,.2f}")
    
    # Получаем текущую цену
    price = connector.get_current_price("XAUUSD")
    if price:
        print(f"\n💰 XAUUSD: ${price:.2f}")
    
    # Получаем историю
    df = connector.get_history(
        symbol="XAUUSD",
        timeframe="H1",
        count=100
    )
    
    if not df.empty:
        print(f"\n📈 История XAUUSD H1: {len(df)} баров")
        print(f"   Период: {df['time'].min()} → {df['time'].max()}")
        print(f"   High: ${df['high'].max():.2f}")
        print(f"   Low: ${df['low'].min():.2f}")
    
    connector.disconnect()
    return df


def demo_mexc():
    """Демо работы с MEXC (публичные данные)"""
    print("\n" + "="*60)
    print("📊 ДЕМО: MEXC Connector (без API ключей)")
    print("="*60)
    
    # MEXC можно использовать без API ключей для получения данных
    connector = MEXCConnector()
    
    if not connector.connect():
        print("❌ Не удалось подключиться к MEXC")
        return None
    
    print("✅ MEXC подключен")
    
    # Список символов
    symbols = connector.get_available_symbols()[:10]
    print(f"\n📋 Доступные символы (первые 10):")
    for s in symbols:
        print(f"   - {s}")
    
    # Получаем текущую цену BTC
    price = connector.get_current_price("BTC/USDT")
    if price:
        print(f"\n💰 BTC/USDT: ${price:,.2f}")
    
    # Получаем историю
    df = connector.get_history(
        symbol="BTC/USDT",
        timeframe="1h",
        count=100
    )
    
    if not df.empty:
        print(f"\n📈 История BTC/USDT 1h: {len(df)} баров")
        print(f"   Период: {df['time'].min()} → {df['time'].max()}")
        print(f"   High: ${df['high'].max():,.2f}")
        print(f"   Low: ${df['low'].min():,.2f}")
    
    connector.disconnect()
    return df


def demo_data_manager():
    """Демо работы с DataManager"""
    print("\n" + "="*60)
    print("📊 ДЕМО: DataManager - Унифицированный доступ")
    print("="*60)
    
    dm = DataManager()
    
    # Пробуем получить данные из разных источников
    sources_to_try = [
        ("mexc", "BTC/USDT", "1h"),  # Крипто всегда работает
        ("mt5", "XAUUSD", "H1"),     # MT5 если установлен
    ]
    
    for source, symbol, tf in sources_to_try:
        print(f"\n🔄 Попытка: {source.upper()} - {symbol} {tf}")
        
        try:
            df = dm.fetch_data(
                source=source,
                symbol=symbol,
                timeframe=tf,
                count=50,
                save=True
            )
            
            if not df.empty:
                print(f"   ✅ Получено {len(df)} баров")
                print(f"   💾 Данные сохранены")
            else:
                print(f"   ⚠️ Нет данных")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Показываем сохранённые данные
    print("\n📂 Доступные сохранённые данные:")
    available = dm.list_available_data()
    for source, symbols in available.items():
        print(f"\n   {source.upper()}:")
        for sym, tfs in symbols.items():
            print(f"      {sym}: {', '.join(tfs)}")
    
    dm.disconnect_all()


def main():
    print("\n" + "🚀"*30)
    print("  GOLDEN BREEZE - MULTI-CONNECTOR SYSTEM DEMO")
    print("🚀"*30)
    
    # Демо MT5
    try:
        demo_mt5()
    except Exception as e:
        print(f"❌ MT5 Demo Error: {e}")
    
    # Демо MEXC
    try:
        demo_mexc()
    except Exception as e:
        print(f"❌ MEXC Demo Error: {e}")
    
    # Демо DataManager
    try:
        demo_data_manager()
    except Exception as e:
        print(f"❌ DataManager Demo Error: {e}")
    
    print("\n" + "="*60)
    print("✅ ДЕМО ЗАВЕРШЕНО")
    print("="*60)
    print("""
Следующие шаги:
1. Настройте credentials в aimodule/config.py
2. Для MEXC: добавьте api_key и api_secret для торговли
3. Для TradeLocker: добавьте email, password и server
4. Используйте DataManager.fetch_training_data() для сбора данных
    """)


if __name__ == "__main__":
    main()

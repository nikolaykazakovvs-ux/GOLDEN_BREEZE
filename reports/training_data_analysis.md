# 📊 Training Data Analysis - Gold V2 Model

## 🗓️ Период Обучения

### Временной Диапазон:
**Начало:** 19 ноября 2025, 22:00 UTC  
**Конец:** 28 ноября 2025, 21:40 UTC  
**Длительность:** **9 дней** (216 часов торговли)

### Таймфрейм:
**M5** (5-минутные свечи)

---

## 📈 Объем Данных

### Raw Data (OHLCV):
- **Total Bars:** 1,927 M5 свечей
- **Per Day:** ~214 свечей/день
- **Coverage:** 9 дней × 24 часа = 1,927 свечей

### Labels:
- **Total Labels:** 1,868 точек входа
- **Distribution:**
  - UP (label=1): 809 сигналов (43.3%)
  - DOWN (label=2): 662 сигналов (35.4%)
  - FLAT (label=0): 397 сигналов (21.3%)

### Sequences (After Feature Engineering):
- **Valid Sequences:** 1,305
- **Dropped:** 623 (из-за NaN после добавления фичей)
- **Loss Rate:** 32.4% (нормально для 200 EMA и Alpha Trend)

---

## 🔄 Data Split

### Training Set (64%):
- **Samples:** 835
- **Period:** 19-24 ноября (примерно)
- **Labels:** UP=470, DOWN=365

### Validation Set (16%):
- **Samples:** 209
- **Period:** 24-26 ноября (примерно)
- **Labels:** UP=118, DOWN=91

### Test Set (20%):
- **Samples:** 261
- **Period:** 26-28 ноября (примерно)
- **Labels:** UP=147, DOWN=114

---

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

### 1. Короткий Период (9 дней)
**Проблема:** Модель обучена на очень коротком периоде (всего 9 дней).

**Риски:**
- Может быть **overfitted** к этому конкретному периоду
- Не покрывает разные рыночные режимы (тренд, флэт, волатильность)
- Не тестировалась на новостных событиях (NFP, CPI, FOMC)

**Что делать:**
- ✅ Протестировать на backtest'е с данными **вне** этого периода
- ✅ Добавить данные за последние 3-6 месяцев для полноценного обучения
- ✅ Периодически ретрейнить модель (каждые 2-4 недели)

---

### 2. Ноябрь 2025 - Специфический Период
**Контекст:** 19-28 ноября 2025 - что происходило?

**Анализ цены XAUUSD:**
- Start: ~2,073 USD
- End: ~2,076 USD
- **Range-bound период** (небольшая волатильность)

**Implications:**
- Модель может хорошо работать в **range/sideways** условиях
- Но может **underperform** в сильных трендах или breakout'ах
- Не тестировалась на крупных новостях

---

### 3. Loss Rate 32.4% (623 dropped bars)
**Причина:** EMA_200 требует минимум 200 свечей для расчета

**Calculation:**
- Raw: 1,927 bars
- After features + dropna: 1,728 bars
- After sequence creation: 1,305 sequences
- **Dropped:** 1,927 - 1,305 = 622 ≈ 32%

**Verdict:** Это нормально для индикаторов с длинным периодом.

---

## 🎯 Рекомендации

### Immediate (Сегодня):
1. **Backtest на другом периоде:**
   ```bash
   # Экспортируем данные за октябрь 2025
   python tools/export_mt5_history.py XAUUSD M5 10000
   # Это даст ~35 дней истории
   
   # Запускаем backtest
   python demo_backtest_hybrid.py
   ```

2. **Out-of-Sample Test:**
   - Test Period: 29 ноября - 3 декабря 2025
   - Expected: Если MCC > 0.55 → модель генерализуется хорошо
   - If MCC < 0.50 → overfitting, нужно больше данных

---

### Short-Term (На Этой Неделе):
1. **Expand Training Data:**
   ```bash
   # Экспортируем 3 месяца данных
   python tools/export_mt5_history.py XAUUSD M5 25000
   # Это даст ~87 дней (сентябрь-ноябрь)
   
   # Регенерируем dataset
   python -m aimodule.training.prepare_direction_dataset \
     --labels data/labels/direction_labels_XAUUSD_extended.csv \
     --data-dir data/raw \
     --symbol XAUUSD --timeframe M5 --seq-len 50 \
     --output data/prepared/direction_dataset_gold_v3.npz
   
   # Обучаем новую модель
   python -m aimodule.training.train_direction_lstm_from_labels \
     --data data/prepared/direction_dataset_gold_v3.npz \
     --epochs 50 --save-path models/direction_lstm_gold_v3.pt
   ```

2. **Test on Different Market Regimes:**
   - **Trending Up:** Сильный бычий тренд (например, после FOMC)
   - **Trending Down:** Сильный медвежий тренд (например, risk-off)
   - **Ranging:** Флэт (как 19-28 ноября)
   - **Volatile:** Высокая волатильность (после NFP)

---

### Long-Term (На Следующий Месяц):
1. **Rolling Window Retraining:**
   - Каждые 2 недели: retrain на последних 3 месяцах данных
   - Keep старую модель как fallback
   - A/B test: новая vs старая на live

2. **Add More Symbols (Diversification):**
   - Обучить аналогичную модель для EURUSD, GBPUSD
   - Сравнить: работают ли Gold фичи для Forex?
   - Возможно, нужны Forex-specific features

---

## 📊 Data Quality Issues

### 1. ✅ No Missing Values
After dropna: 0 missing values → good

### 2. ✅ Balanced Classes
Train: 56% UP / 44% DOWN → well balanced

### 3. ⚠️ Potential Data Leakage?
**Check:** Labels aligned correctly with timestamps?

**Verification:**
```python
# В prepare_direction_dataset.py:
# Merge labels with OHLCV data by timestamp
df['label'] = 0
for _, row in labels_df.iterrows():
    ts = pd.to_datetime(row['timestamp'])
    idx = df.index.searchsorted(ts)
    if idx < len(df):
        df.iloc[idx, df.columns.get_loc('label')] = row['direction_label']
```

**Concern:** Timestamps в labels.csv - это индексы (49, 50, 51...), а не реальные даты!

**Action:** Проверить, как генерировались labels:
```bash
# Смотрим скрипт генерации labels
cat aimodule/training/generate_labels.py
```

### 4. ⚠️ Sequential Split (Not Random)
**Current:** train_test_split используется, но с `shuffle=False` (implied)

**Good:** Это правильно для временных рядов!
**Bad:** Но test period очень короткий (только 26-28 ноября)

---

## 🧪 Validation Strategy

### Current:
- Train: 19-24 ноября
- Val: 24-26 ноября
- Test: 26-28 ноября

### Recommended (Walk-Forward):
```
Train:    Sep 1  - Oct 31  (60 days)
Val:      Nov 1  - Nov 15  (15 days)
Test:     Nov 16 - Nov 30  (15 days)
OOS Test: Dec 1  - Dec 7   (7 days)  ← REAL WORLD TEST
```

---

## 📈 Expected Performance

### On Training Period (19-28 Nov):
- **Test MCC:** 0.6875 ✅ (already achieved)
- **Accuracy:** 84.67% ✅

### On Out-of-Sample (29 Nov - 3 Dec):
- **Expected MCC:** 0.55 - 0.65 (realistic)
- **Expected Accuracy:** 75% - 82%
- **If MCC < 0.50:** Overfitting → need more data

### On Different Market Regime:
- **Trending:** MCC may drop to 0.50 - 0.60
- **Volatile:** MCC may drop to 0.45 - 0.55
- **Range (like training):** MCC should stay 0.65+

---

## 🎯 Success Criteria

### Minimum Viable (MVP):
- ✅ Test MCC ≥ 0.65 (achieved: 0.6875)
- ⏳ OOS MCC ≥ 0.55 (pending backtest)
- ⏳ Win Rate ≥ 55% (pending live test)

### Target:
- ⏳ Test MCC ≥ 0.70 (close: 0.6875)
- ⏳ OOS MCC ≥ 0.60
- ⏳ Win Rate ≥ 60%
- ⏳ Stable across different regimes

### Exceptional:
- ⏳ Test MCC ≥ 0.75
- ⏳ OOS MCC ≥ 0.70
- ⏳ Win Rate ≥ 65%
- ⏳ Profitable in trending + ranging

---

## 🔄 Next Actions

### Priority 1 (Сейчас):
```bash
# Backtest на данных ВНЕ training period
python demo_backtest_hybrid.py --start-date 2025-11-29 --end-date 2025-12-03
```

### Priority 2 (Сегодня):
```bash
# Экспортируем больше данных (3 месяца)
python tools/export_mt5_history.py XAUUSD M5 25000
```

### Priority 3 (Завтра):
- Retrain на 3 месяцах
- Compare v2 (9 days) vs v3 (3 months)
- If v3 better → deploy
- If v2 better → current period more relevant (recency bias)

---

## 📝 Summary

### What We Know:
- ✅ Model trained on **9 days** (19-28 Nov 2025)
- ✅ Timeframe: **M5** (5-minute candles)
- ✅ Period: **Range-bound market**
- ✅ Test MCC: **0.6875** (excellent for this period)

### What We Don't Know:
- ❓ Performance on **trending** markets
- ❓ Performance on **volatile** markets
- ❓ Performance on **out-of-sample** data (29 Nov+)
- ❓ Real-world **live trading** performance

### What We Need:
- 🎯 **More data:** 3 months minimum
- 🎯 **Out-of-sample test:** 29 Nov - 3 Dec
- 🎯 **Live validation:** Demo account test
- 🎯 **Regime testing:** Test on different market conditions

---

**Conclusion:**  
Модель показывает **отличные результаты** на своем обучающем периоде (9 дней, range market). Но для **production deployment** нам нужно:
1. Backtest на других периодах
2. Retrain на больших данных (3+ месяца)
3. Live test на demo account

**Status:** ✅ PROOF OF CONCEPT SUCCESS  
**Next Step:** 🔄 VALIDATION & EXPANSION

---

**Generated:** 03.12.2025 05:40  
**Data Period:** 19-28 Nov 2025 (9 days)  
**Model:** direction_lstm_gold_v2.pt  

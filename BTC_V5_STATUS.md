# Статус тренировки BTC v5 Golden Breeze

## Обученная модель
- **Модель**: GoldenBreezeV5Ultimate (327,751 параметров)
- **Эпоха**: 34 из 200
- **Лучший Val MCC**: +0.2595
- **Путь**: `models/v5_btc/best_model.pt`
- **Данные**: BTC/USDT (Binance, M5+H1 bars)

## Метрики по эпохам (сохраненные)
```
Epoch 1:  Val MCC +0.1332
Epoch 10: Val MCC +0.1853
Epoch 20: Val MCC +0.2146
Epoch 30: Val MCC +0.2467
Epoch 34: Val MCC +0.2595 ✨ BEST
```

## Оптимизации реализованы ✅

### GPU Acceleration
- ✅ TF32 включен (`torch.backends.cuda.matmul.allow_tf32`)
- ✅ cuDNN benchmark включен (`torch.backends.cudnn.benchmark`)
- ✅ High precision float32 matmul (`torch.set_float32_matmul_precision('high')`)
- ✅ CUDA non-blocking transfers (`non_blocking=True`)

### Data Pipeline  
- ✅ Batch size: 1024 (default, можно до 1536 на RTX 3070)
- ✅ Num workers: 8 (параллельная загрузка данных)
- ✅ Prefetch factor: 4 (pipelined data feeding)
- ✅ Pin memory: True (pinned CPU→GPU transfers)
- ✅ Persistent workers: True (воркеры между эпохами)

### Model Training
- ✅ Mixed precision (AMP, FP16 + FP32)
- ✅ Gradient scaling (GradScaler)
- ✅ Auto-resume из best_model.pt
- ✅ Auto-backup перед перезапуском
- ✅ Cosine learning rate schedule + warmup

## Как использовать сохраненную модель

### Продолжить обучение со скоростью
```bash
# С оптимальными параметрами (batch 1024, 8 workers)
python train_v5_btc.py --batch-size 1024 --num-workers 8 --epochs 200

# Или если нужен меньший batch (для конкретного GPU):
python train_v5_btc.py --batch-size 512 --num-workers 4
```

**Автоматически**:
- Создаст backup текущего best_model.pt
- Загрузит существующий checkpoint (epoch 34)
- Продолжит с эпохи 35

### Оценить модель на тесте
```bash
python evaluate_best_model.py
```

Выведет:
- Test Loss, Accuracy, MCC
- Per-class accuracy
- Confusion matrix
- JSON report в `reports/v5_btc_evaluation.json`

## Ресурсы при обучении
- GPU: RTX 3070 (8 GB VRAM)
- CPU: ~50% load при batch 1024, 8 workers
- RAM: ~85% (data в памяти)
- GPU VRAM: ~69-74% (5.5/8 GB)
- GPU utilization: 70-85% (оптимизировано)

## Проблемы и решения

### ❌ KeyboardInterrupt при чтении npz
Если возникает при запуске - проверьте:
1. Файл `data/prepared/btc_v5.npz` целый (проверь размер ~112 MB)
2. Достаточно свободного места на диске
3. Дай больше памяти (уменьши batch size или num_workers)

### ❌ CUDA OOM (Out of Memory)
Решение:
```bash
# Уменьши batch size
python train_v5_btc.py --batch-size 768
# или
python train_v5_btc.py --batch-size 512

# Уменьши число воркеров
python train_v5_btc.py --batch-size 1024 --num-workers 4
```

## Следующие шаги для ускорения обучения

1. **Увеличить GPU load дальше**:
   - `--batch-size 1536` (если влезет) + `--num-workers 10`
   - `--prefetch-factor 6` для более агрессивного pipelining

2. **Использовать gradient accumulation**:
   - Если нужен ещё больший effective batch (но slower)

3. **Distributed training** (если есть несколько GPU):
   - Использовать `torch.nn.DataParallel` или `DistributedDataParallel`

## Файлы
- `train_v5_btc.py` - основной скрипт обучения
- `evaluate_best_model.py` - оценка на тесте
- `tools/fetch_crypto_history.py` - загрузка данных (Binance)
- `tools/precompute_btc.py` - подготовка данных (features, labels)
- `models/v5_btc/best_model.pt` - лучший checkpoint (epoch 34)
- `models/v5_btc/best_model_backup_*.pt` - автоматические бэкапы

---
**Дата**: 2025-12-06
**Ветка**: fusion-transformer-v4
**Модель**: GoldenBreezeV5Ultimate_BTC

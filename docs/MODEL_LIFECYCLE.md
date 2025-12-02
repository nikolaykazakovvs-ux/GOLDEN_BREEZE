# Model Lifecycle (v3.0)

Covers regime clustering and direction LSTM models: training, saving, loading, and inference.

## Paths
- `REGIME_ML_MODEL_PATH`: sklearn Pipeline with scaler + clustering (kmeans/gmm)
- `DIRECTION_LSTM_MODEL_PATH`: PyTorch state dict for LSTM + meta.json sidecar

## Training
### Regime
- Use `aimodule/training/train_regime_cluster.py` with CSV history.
- Features include log_returns, volatility, RSI, SMA slope, price position, volume dynamics.

### Direction
- Use `aimodule/training/train_direction_model.py`.
- Wrapper computes labels via epsilon bands; tracks metrics (Accuracy, F1, MCC); early stop.

## Inference
- `aimodule/inference/predict_direction.py` loads LSTM; falls back to `DirectionPredictor` if missing.
- Sentiment Engine blends HF/lexicon/regime baseline.

## Versioning
- Sidecar `meta.json` includes training config and metrics; bump app version to v3.0 in server `/health`.


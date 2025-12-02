# FEEDBACK Protocol (v3.0)

This document specifies the format and semantics of feedback records used by Golden Breeze v3.0 Self-Learning layer.

## Record Fields
- symbol: string (e.g., BTCUSDT)
- regime: string (e.g., trend, range, volatile)
- predicted_direction: string (up|down|flat)
- confidence: float [0,1]
- entry_price: float
- exit_price: float
- timestamp: ISO8601 string (UTC)
- actual_outcome: derived boolean by store (True if prediction aligns with PnL > 0)

## Storage
- CSV file at path configured by `FEEDBACK_CSV_PATH` in `aimodule.config`.
- Append-only; periodically pruned using `clear_old_records()`.

## Online Updates
- Thresholds updated via `OnlineUpdater.update_thresholds(store)`:
  - min_confidence_by_regime: raised if win rate < target; lowered if win rate > target.
  - sentiment_skip_threshold: raised if sentiment-negative correlates with losses.
- Persisted JSON at `DYNAMIC_CONFIG_PATH`.

## API Contract
- POST /feedback accepts JSON conforming to the fields above; returns status and updated config snapshot.


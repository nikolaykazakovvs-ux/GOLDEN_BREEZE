# Migration Guide: v2.0 → v3.0

## Overview
Golden Breeze v3.0 introduces modular ML upgrades, sentiment engine v3, self-learning, GPU support, and an AI connector client, while keeping the trading strategy external and API contracts stable.

## Key Changes
- New `/feedback` endpoint; CSV FeedbackStore; OnlineUpdater dynamic thresholds.
- Regime model pipeline saved as sklearn Pipeline with scaler.
- Direction model upgraded to LSTM with early stopping and metrics.
- Sentiment engine with TTL-cached news source prioritized HF → lexicon → regime baseline.
- Config centralizes paths and CUDA device detection.

## What stays the same
- External strategy remains separate.
- `/predict` response structure preserved, with additional fields optional.

## Steps
1. Install new dependencies (PyTorch, scikit-learn, FastAPI, transformers optional).
2. Train regime and direction models; confirm saved paths.
3. Start local AI gateway; verify `/health` returns v3.0.
4. Integrate `GoldenBreezeClient` for predictions and feedback.


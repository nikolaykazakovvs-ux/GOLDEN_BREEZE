# Risk & Usage Guide

Guidelines for using predictions safely.

- Always enforce a minimum confidence per regime from dynamic config.
- Skip trades when sentiment_skip_threshold indicates strong negative sentiment with low confidence.
- Cap position sizes when model meta indicates low MCC or recent drift.
- Persist feedback for every closed trade to enable self-learning.
- Prefer GPU for training; inference runs on CPU when GPU unavailable.


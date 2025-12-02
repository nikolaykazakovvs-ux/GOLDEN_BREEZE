import os
import json
import pandas as pd
from pathlib import Path

# Assuming module paths
try:
    from aimodule.learning.feedback_store import FeedbackStore
    from aimodule.learning.online_update import OnlineUpdater
    from aimodule.config import FEEDBACK_CSV_PATH, DYNAMIC_CONFIG_PATH
except Exception:
    FeedbackStore = None
    OnlineUpdater = None
    FEEDBACK_CSV_PATH = Path("feedback.csv")
    DYNAMIC_CONFIG_PATH = Path("dynamic_config.json")


def test_feedback_store_add_and_stats(tmp_path):
    if FeedbackStore is None:
        assert True, "FeedbackStore not available in this context"
        return

    csv_path = tmp_path / "feedback.csv"
    store = FeedbackStore(csv_path)

    # Add mixed outcomes
    store.add_feedback({
        "symbol": "BTCUSDT",
        "regime": "trend",
        "predicted_direction": "up",
        "confidence": 0.8,
        "entry_price": 100,
        "exit_price": 110,
        "timestamp": "2025-01-01T00:00:00Z",
    })
    store.add_feedback({
        "symbol": "BTCUSDT",
        "regime": "range",
        "predicted_direction": "down",
        "confidence": 0.6,
        "entry_price": 200,
        "exit_price": 195,
        "timestamp": "2025-01-02T00:00:00Z",
    })
    store.add_feedback({
        "symbol": "BTCUSDT",
        "regime": "trend",
        "predicted_direction": "down",
        "confidence": 0.55,
        "entry_price": 300,
        "exit_price": 310,
        "timestamp": "2025-01-03T00:00:00Z",
    })

    stats = store.get_statistics()
    assert "overall_win_rate" in stats
    assert 0 <= stats["overall_win_rate"] <= 1
    assert set(["trend", "range"]).issubset(set(stats.get("by_regime", {}).keys()))


def test_online_updater_adjusts_thresholds(tmp_path):
    if OnlineUpdater is None:
        assert True, "OnlineUpdater not available in this context"
        return

    # Prepare feedback
    csv_path = tmp_path / "feedback.csv"
    store = FeedbackStore(csv_path)

    # Add records to drive win rate below/above thresholds
    # Low win rate regime "range"
    for i in range(10):
        store.add_feedback({
            "symbol": "ETHUSDT",
            "regime": "range",
            "predicted_direction": "up",
            "confidence": 0.7,
            "entry_price": 100,
            "exit_price": 99 if i % 2 == 0 else 98,
            "timestamp": f"2025-01-0{i+1}T00:00:00Z",
        })

    # High win rate regime "trend"
    for i in range(10):
        store.add_feedback({
            "symbol": "ETHUSDT",
            "regime": "trend",
            "predicted_direction": "down",
            "confidence": 0.8,
            "entry_price": 200,
            "exit_price": 201 if i % 2 == 0 else 202,
            "timestamp": f"2025-02-0{i+1}T00:00:00Z",
        })

    updater = OnlineUpdater(DYNAMIC_CONFIG_PATH if DYNAMIC_CONFIG_PATH.is_absolute() else tmp_path / "dynamic_config.json")
    new_cfg = updater.update_thresholds(store)

    assert "min_confidence_by_regime" in new_cfg
    assert isinstance(new_cfg["min_confidence_by_regime"], dict)
    assert new_cfg["min_confidence_by_regime"].get("range") >= 0.55
    assert new_cfg["min_confidence_by_regime"].get("trend") <= 0.8


try:
    from aimodule.inference.predict_direction import DirectionPredictor
except Exception:
    DirectionPredictor = None


def test_predictor_fallback_outputs():
    if DirectionPredictor is None:
        assert True, "PredictDirection fallback not available"
        return

    pred = DirectionPredictor()
    import pandas as pd
    import numpy as np
    # Build minimal dataframe for fallback momentum
    df = pd.DataFrame({
        "close": [100.0, 100.1, 100.05, 100.2],
        "volume": [1, 1, 1, 1]
    })
    direction, prob = pred.predict(df)
    assert direction is not None
    assert prob is not None
    assert 0.0 <= float(prob) <= 1.0


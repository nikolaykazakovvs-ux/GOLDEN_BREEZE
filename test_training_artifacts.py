"""
Check training artifacts after run_full_training_gpu.ps1.
"""
import sys
from pathlib import Path
import json

def main():
    root = Path(__file__).resolve().parent
    symbol = "XAUUSD"

    labels = list((root / "data/labels").glob(f"direction_labels_{symbol}*.csv"))
    dataset = list((root / "data/prepared").glob(f"direction_dataset_{symbol}*.npz"))
    model = root / f"models/direction_lstm_hybrid_{symbol}.pt"
    reports = list((root / "reports").glob(f"hybrid_v1.1_{symbol}_*.md"))
    meta_json = model.with_suffix('.json')

    ok = True
    if labels:
        print(f"OK: labels found ({labels[0].name})")
    else:
        print("ERROR: labels not found in data/labels/")
        ok = False

    if dataset:
        print(f"OK: dataset found ({dataset[0].name})")
    else:
        print("ERROR: dataset not found in data/prepared/")
        ok = False

    if model.exists():
        print(f"OK: model found ({model.name})")
    else:
        print("ERROR: model .pt not found in models/")
        ok = False

    if reports:
        print(f"OK: report found ({reports[0].name})")
    else:
        print("ERROR: report not found in reports/")
        ok = False

    if meta_json.exists():
        try:
            meta = json.load(open(meta_json, 'r'))
            required = ["test_metrics", "seq_len"]
            missing = [k for k in required if k not in meta]
            if missing:
                print(f"ERROR: metadata missing fields: {missing}")
                ok = False
            else:
                tm = meta.get("test_metrics", {})
                keys = ["accuracy", "f1_macro", "mcc"]
                miss2 = [k for k in keys if k not in tm]
                if miss2:
                    print(f"ERROR: test_metrics missing: {miss2}")
                    ok = False
                else:
                    print("OK: metadata contains accuracy, f1_macro, mcc, seq_len")
        except Exception as e:
            print(f"ERROR: reading metadata failed: {e}")
            ok = False
    else:
        print("ERROR: metadata JSON not found next to model")
        ok = False

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

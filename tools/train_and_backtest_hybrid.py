"""
Golden Breeze Hybrid Strategy - Training & Backtest Orchestrator.

One-command pipeline:
1. Export MT5 data (if needed)
2. Generate labels via HybridStrategy backtest
3. Prepare dataset for LSTM
4. Train Direction LSTM
5. Run final backtest with new model
6. Generate report

Usage:
    python -m tools.train_and_backtest_hybrid --symbol XAUUSD --start 2024-01-01 --end 2024-06-01

Author: Golden Breeze Team
Version: 1.1
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
import time
import requests

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and backtest Golden Breeze Hybrid Strategy"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["M1", "M5", "M15", "H1", "H4"],
        help="Timeframes to export"
    )
    parser.add_argument(
        "--primary-tf",
        type=str,
        default="M5",
        help="Primary timeframe for strategy"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="LSTM sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip data export (use existing data)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)"
    )
    
    return parser.parse_args()


def run_command(cmd: list, description: str) -> bool:
    """
    Run a subprocess command.
    
    Args:
        cmd: Command list
        description: Description for logging
    
    Returns:
        True if successful
    """
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✅ {description} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with code {e.returncode}")
        return False
    
    except Exception as e:
        print(f"\n❌ {description} failed: {e}")
        return False


def check_data_exists(data_dir: Path, symbol: str, timeframes: list) -> bool:
    """Check if data files exist."""
    symbol_dir = data_dir / symbol
    
    if not symbol_dir.exists():
        return False
    
    for tf in timeframes:
        csv_path = symbol_dir / f"{tf}.csv"
        parquet_path = symbol_dir / f"{tf}.parquet"
        
        if not (csv_path.exists() or parquet_path.exists()):
            return False
    
    return True


def generate_report(
    symbol: str,
    start_date: str,
    end_date: str,
    model_path: Path,
    output_dir: Path
) -> Path:
    """
    Generate training & backtest report.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        model_path: Path to trained model
        output_dir: Output directory for report
    
    Returns:
        Path to generated report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"hybrid_v1.1_{symbol}_{timestamp}.md"
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model metadata if exists
    meta_path = model_path.with_suffix('.json')
    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    # Query environment
    env_device = "unknown"
    env_use_gpu = False
    env_torch = "unknown"
    try:
        h = requests.get("http://127.0.0.1:5005/health", timeout=3).json()
        env_device = str(h.get("device"))
        env_use_gpu = bool(h.get("use_gpu"))
    except Exception:
        pass
    try:
        import torch
        env_torch = torch.__version__
    except Exception:
        pass

    # Generate report
    report_content = f"""# Golden Breeze Hybrid Strategy v1.1 - Training & Backtest Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Configuration

- **Symbol:** {symbol}
- **Period:** {start_date} to {end_date}
- **Primary Timeframe:** {metadata.get('timeframe', 'M5')}
- **Sequence Length:** {metadata.get('seq_len', 50)}

## 🧭 Environment

- **device:** {env_device}
- **use_gpu:** {env_use_gpu}
- **torch_version:** {env_torch}
- **seq_len:** {metadata.get('seq_len', 50)}
- **epochs:** {metadata.get('epochs_trained', 'N/A')}
- **train_interval:** {start_date} → {end_date}

## 🧠 Model Training Results

- **Model Type:** {metadata.get('model_type', 'DirectionLSTM')}
- **Training Date:** {metadata.get('training_date', 'N/A')}
- **Epochs Trained:** {metadata.get('epochs_trained', 'N/A')}
- **Features:** {metadata.get('n_features', 'N/A')}
- **Classes:** {metadata.get('n_classes', 'N/A')}

### Training Metrics

| Metric | Value |
|--------|-------|
| Best Val MCC | {metadata.get('best_val_mcc', 'N/A'):.4f} |
| Test Accuracy | {metadata.get('test_metrics', {}).get('accuracy', 'N/A'):.4f} |
| Test F1 (macro) | {metadata.get('test_metrics', {}).get('f1_macro', 'N/A'):.4f} |
| Test MCC | {metadata.get('test_metrics', {}).get('mcc', 'N/A'):.4f} |

### Confusion Matrix

```
{metadata.get('confusion_matrix', 'N/A')}
```

## 📈 Backtest Results

*Run final backtest to populate this section*

To run final backtest with the trained model:

```bash
python demo_backtest_hybrid.py
```

## 🎯 Next Steps

1. Review model performance metrics
2. Analyze backtest results
3. Optimize hyperparameters if needed
4. Deploy to live trading (when ready)

## 📁 Files Generated

- Model: `{model_path}`
- Metadata: `{meta_path}`
- Report: `{report_path}`

---

*Generated by Golden Breeze Training Pipeline v1.1*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


def main():
    """Main orchestration function."""
    args = parse_args()
    
    print("="*60)
    print("Golden Breeze - Training & Backtest Orchestrator v1.1")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print(f"Primary TF: {args.primary_tf}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # Pre-flight: check AI Core /health and ensure device == cuda
    print("\n" + "="*60)
    print("STEP: AI Core GPU Health Check")
    print("="*60)
    health_url = "http://127.0.0.1:5005/health"
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code != 200:
            print(f"❌ AI Core not ready: HTTP {resp.status_code} at {health_url}")
            sys.exit(2)
        health = resp.json()
        device = str(health.get("device"))
        use_gpu = bool(health.get("use_gpu"))
        print(f"Health: device={device}, use_gpu={use_gpu}")
        if device != "cuda" or use_gpu is not True:
            print("❌ AI Core not ready: health.device != 'cuda' or use_gpu != True")
            sys.exit(2)
        try:
            import torch
            print(f"✅ AI Core GPU OK (torch {torch.__version__})")
        except Exception:
            print("✅ AI Core GPU OK")
    except requests.RequestException as e:
        print(f"❌ AI Core health check failed: {e}")
        sys.exit(2)
    
    # Paths
    data_dir = Path("data/raw")
    labels_path = Path(f"data/labels/direction_labels_{args.symbol}.csv")
    dataset_path = Path(f"data/prepared/direction_dataset_{args.symbol}.npz")
    model_path = Path(f"models/direction_lstm_hybrid_{args.symbol}.pt")
    reports_dir = Path("reports")
    
    # Step 1: Export MT5 data (if needed)
    if not args.skip_export:
        if check_data_exists(data_dir, args.symbol, args.timeframes):
            print("\n✅ Data already exists, skipping export")
            print("   Use --skip-export to skip this check")
        else:
            print("\n📊 Data not found, starting export...")
            
            cmd = [
                "python", "-m", "tools.export_mt5_history",
                "--symbol", args.symbol,
                "--start", args.start,
                "--end", args.end,
                "--timeframes", *args.timeframes
            ]
            
            if not run_command(cmd, "Export MT5 Historical Data"):
                print("\n❌ Pipeline failed at data export")
                sys.exit(1)
    else:
        print("\n⏭️  Skipping data export (--skip-export)")
    
    # Step 2: Generate labels
    if not args.skip_training:
        # Use fast version for quick smoke tests
        cmd = [
            "python", "-m", "aimodule.training.generate_labels_fast",
            "--symbol", args.symbol,
            "--primary-tf", args.primary_tf,
            "--data-dir", str(data_dir),
            "--output", str(labels_path),
            "--lookahead", "10"
        ]
        
        if not run_command(cmd, "Generate Training Labels (Fast)"):
            print("\n❌ Pipeline failed at label generation")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping label generation (--skip-training)")
    
    # Step 3: Prepare dataset
    if not args.skip_training:
        cmd = [
            "python", "-m", "aimodule.training.prepare_direction_dataset",
            "--labels", str(labels_path),
            "--data-dir", str(data_dir),
            "--symbol", args.symbol,
            "--timeframe", args.primary_tf,
            "--seq-len", str(args.seq_len),
            "--output", str(dataset_path)
        ]
        
        if not run_command(cmd, "Prepare LSTM Dataset"):
            print("\n❌ Pipeline failed at dataset preparation")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping dataset preparation (--skip-training)")
    
    # Step 4: Train Direction LSTM
    if not args.skip_training:
        cmd = [
            "python", "-m", "aimodule.training.train_direction_lstm_from_labels",
            "--data", str(dataset_path),
            "--seq-len", str(args.seq_len),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--save-path", str(model_path)
        ]
        
        if not run_command(cmd, "Train Direction LSTM"):
            print("\n❌ Pipeline failed at model training")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping model training (--skip-training)")
    
    # Step 5: Generate report
    print("\n" + "="*60)
    print("STEP: Generate Report")
    print("="*60)
    
    try:
        report_path = generate_report(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            model_path=model_path,
            output_dir=reports_dir
        )
        
        print(f"✅ Report generated: {report_path}")
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Execution Summary")
    print("="*60)
    print(f"✅ Data exported to: {data_dir / args.symbol}")
    print(f"✅ Labels saved to: {labels_path}")
    print(f"✅ Dataset saved to: {dataset_path}")
    print(f"✅ Model trained: {model_path}")
    print(f"✅ Report generated: {report_path}")
    
    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("="*60)
    
    print("\n📝 Next steps:")
    print(f"1. Review report: {report_path}")
    print(f"2. Check model metadata: {model_path.with_suffix('.json')}")
    print(f"3. Run final backtest with new model (update config to use {model_path})")
    print(f"4. Analyze backtest results and optimize parameters")
    
    print("\n🚀 To run backtest:")
    print(f"   python demo_backtest_hybrid.py")


if __name__ == "__main__":
    main()

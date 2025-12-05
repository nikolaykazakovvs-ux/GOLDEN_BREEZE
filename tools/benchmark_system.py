"""
Golden Breeze - Hardware Benchmark Script

Empirically determines optimal training parameters for current hardware:
- Maximum safe batch size
- VRAM usage profile
- FP16 vs FP32 performance
- CPU/DataLoader bottleneck detection

Hardware Target: Ryzen 2700 + RTX 3070 8GB

Author: Golden Breeze Team
Date: 2025-12-05
"""

import os
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from aimodule.models.v4_lstm import LSTMModelV4, LSTMConfig


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    else:
        allocated = reserved = max_allocated = 0.0
    
    ram_used = psutil.virtual_memory().used / 1024**3
    
    return {
        'vram_allocated': allocated,
        'vram_reserved': reserved,
        'vram_max': max_allocated,
        'ram_used': ram_used,
    }


def clear_memory():
    """Clear CUDA cache and reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def generate_dummy_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate dummy data matching real dataset dimensions.
    
    Returns:
        x_fast: (batch, 50, 15) - M5 sequences
        x_slow: (batch, 20, 8) - H1 sequences
        x_strat: (batch, 64) - Strategy features
        labels: (batch,) - 3-class labels
    """
    x_fast = torch.randn(batch_size, 50, 15)
    x_slow = torch.randn(batch_size, 20, 8)
    x_strat = torch.randn(batch_size, 64)
    labels = torch.randint(0, 3, (batch_size,))
    
    return x_fast, x_slow, x_strat, labels


def test_batch_size(
    model: nn.Module,
    batch_size: int,
    device: str,
    use_amp: bool = False,
    num_iterations: int = 10,
) -> Dict[str, any]:
    """
    Test a specific batch size.
    
    Returns:
        dict with status, vram_peak, time_per_batch
    """
    clear_memory()
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None
    
    try:
        times = []
        
        for i in range(num_iterations):
            # Generate batch
            x_fast, x_slow, x_strat, labels = generate_dummy_batch(batch_size)
            x_fast = x_fast.to(device)
            x_slow = x_slow.to(device)
            x_strat = x_strat.to(device)
            labels = labels.to(device)
            
            start_time = time.time()
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    logits = model(x_fast, x_slow, x_strat)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_fast, x_slow, x_strat)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Skip first iteration (warmup)
            if i == 0:
                times.clear()
        
        mem_info = get_memory_info()
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            'status': 'OK',
            'vram_peak_gb': mem_info['vram_max'],
            'time_per_batch_ms': avg_time * 1000,
            'samples_per_sec': batch_size / avg_time if avg_time > 0 else 0,
        }
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return {
                'status': 'OOM',
                'vram_peak_gb': get_memory_info()['vram_max'],
                'time_per_batch_ms': None,
                'samples_per_sec': None,
            }
        else:
            return {
                'status': f'ERROR: {str(e)[:50]}',
                'vram_peak_gb': None,
                'time_per_batch_ms': None,
                'samples_per_sec': None,
            }
    
    except Exception as e:
        return {
            'status': f'ERROR: {str(e)[:50]}',
            'vram_peak_gb': None,
            'time_per_batch_ms': None,
            'samples_per_sec': None,
        }


def run_benchmark():
    """Run full hardware benchmark."""
    
    print("=" * 80)
    print("🔥 GOLDEN BREEZE - HARDWARE BENCHMARK")
    print("=" * 80)
    print()
    
    # System info
    print("📊 System Information:")
    print(f"   CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = 'cuda'
    else:
        print("   GPU: Not available")
        device = 'cpu'
    
    print()
    
    # Initialize model
    print("🔧 Initializing LSTM V4 model...")
    config = LSTMConfig()
    model = LSTMModelV4(config)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print()
    
    # ========================================================================
    # TEST 1: Batch Size Sweep (FP32)
    # ========================================================================
    print("=" * 80)
    print("TEST 1: Batch Size Sweep (FP32 Precision)")
    print("=" * 80)
    print()
    
    batch_sizes = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    results_fp32 = []
    max_safe_batch = 16
    
    print(f"{'Batch':>6} | {'Status':^10} | {'VRAM (GB)':>10} | {'Time/Batch (ms)':>16} | {'Samples/sec':>12}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        result = test_batch_size(model, batch_size, device, use_amp=False, num_iterations=5)
        results_fp32.append((batch_size, result))
        
        status = result['status']
        vram = f"{result['vram_peak_gb']:.2f}" if result['vram_peak_gb'] else "N/A"
        time_ms = f"{result['time_per_batch_ms']:.1f}" if result['time_per_batch_ms'] else "N/A"
        sps = f"{result['samples_per_sec']:.1f}" if result['samples_per_sec'] else "N/A"
        
        print(f"{batch_size:>6} | {status:^10} | {vram:>10} | {time_ms:>16} | {sps:>12}")
        
        if result['status'] == 'OK':
            max_safe_batch = batch_size
        elif result['status'] == 'OOM':
            print(f"\n⚠️  OOM detected at batch={batch_size}. Max safe batch (FP32): {max_safe_batch}")
            break
    
    print()
    
    # ========================================================================
    # TEST 2: FP16 (Mixed Precision) at Safe Batch Size
    # ========================================================================
    if device == 'cuda' and max_safe_batch >= 32:
        print("=" * 80)
        print("TEST 2: Mixed Precision (FP16) Performance")
        print("=" * 80)
        print()
        
        print(f"Testing at batch size: {max_safe_batch}")
        print()
        
        # FP32 baseline
        result_fp32 = test_batch_size(model, max_safe_batch, device, use_amp=False, num_iterations=10)
        
        # FP16 test
        result_fp16 = test_batch_size(model, max_safe_batch, device, use_amp=True, num_iterations=10)
        
        print(f"{'Precision':>10} | {'VRAM (GB)':>10} | {'Time/Batch (ms)':>16} | {'Samples/sec':>12}")
        print("-" * 80)
        
        for name, result in [('FP32', result_fp32), ('FP16 (AMP)', result_fp16)]:
            vram = f"{result['vram_peak_gb']:.2f}" if result['vram_peak_gb'] else "N/A"
            time_ms = f"{result['time_per_batch_ms']:.1f}" if result['time_per_batch_ms'] else "N/A"
            sps = f"{result['samples_per_sec']:.1f}" if result['samples_per_sec'] else "N/A"
            print(f"{name:>10} | {vram:>10} | {time_ms:>16} | {sps:>12}")
        
        if result_fp16['time_per_batch_ms'] and result_fp32['time_per_batch_ms']:
            speedup = result_fp32['time_per_batch_ms'] / result_fp16['time_per_batch_ms']
            print(f"\n⚡ FP16 Speedup: {speedup:.2f}x")
        
        print()
        
        # Try larger batch with FP16
        print("Testing larger batch sizes with FP16...")
        print()
        
        larger_batches = [b for b in [max_safe_batch * 2, max_safe_batch * 3, max_safe_batch * 4] if b <= 512]
        max_safe_batch_fp16 = max_safe_batch
        
        print(f"{'Batch':>6} | {'Status':^10} | {'VRAM (GB)':>10} | {'Time/Batch (ms)':>16} | {'Samples/sec':>12}")
        print("-" * 80)
        
        for batch_size in larger_batches:
            result = test_batch_size(model, batch_size, device, use_amp=True, num_iterations=5)
            
            status = result['status']
            vram = f"{result['vram_peak_gb']:.2f}" if result['vram_peak_gb'] else "N/A"
            time_ms = f"{result['time_per_batch_ms']:.1f}" if result['time_per_batch_ms'] else "N/A"
            sps = f"{result['samples_per_sec']:.1f}" if result['samples_per_sec'] else "N/A"
            
            print(f"{batch_size:>6} | {status:^10} | {vram:>10} | {time_ms:>16} | {sps:>12}")
            
            if result['status'] == 'OK':
                max_safe_batch_fp16 = batch_size
            elif result['status'] == 'OOM':
                print(f"\n⚠️  OOM detected at batch={batch_size}. Max safe batch (FP16): {max_safe_batch_fp16}")
                break
        
        print()
    
    # ========================================================================
    # FINAL RECOMMENDATIONS
    # ========================================================================
    print("=" * 80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if device == 'cpu':
        print("⚠️  GPU not available. Training on CPU will be very slow.")
        print(f"   Recommended Batch Size: 16-32")
    else:
        print(f"🔥 Max Safe Batch Size (FP32): {max_safe_batch}")
        
        if max_safe_batch >= 32:
            # Check if FP16 gives speedup
            use_fp16 = False
            if 'result_fp16' in locals() and result_fp16['status'] == 'OK':
                if result_fp16['time_per_batch_ms'] < result_fp32['time_per_batch_ms'] * 0.9:
                    use_fp16 = True
                    print(f"⚡ FP16 Recommended: YES (faster + more VRAM)")
                    print(f"   Max Safe Batch Size (FP16): {max_safe_batch_fp16}")
                else:
                    print(f"⚠️  FP16 Recommended: NO (no significant speedup on RTX 3070)")
            
            # Conservative recommendation (80% of max)
            recommended_batch = int(max_safe_batch * 0.8)
            recommended_batch = max(16, (recommended_batch // 16) * 16)  # Round to multiple of 16
            
            print()
            print("=" * 80)
            print("💡 OPTIMAL TRAINING CONFIG:")
            print("=" * 80)
            print(f"   Batch Size: {recommended_batch}")
            print(f"   Precision: {'FP16 (Mixed Precision)' if use_fp16 else 'FP32'}")
            print(f"   DataLoader Workers: 4-6 (Ryzen 2700 has 8 cores)")
            print(f"   Pin Memory: True")
            print()
            print("   Training Command:")
            print(f"   python -m aimodule.training.train_v4_lstm \\")
            print(f"     --data-path data/prepared/v4_6year_dataset.npz \\")
            print(f"     --batch-size {recommended_batch} \\")
            print(f"     --epochs 500 \\")
            print(f"     --patience 50 \\")
            print(f"     --save-dir models/v4_6year")
            
            if use_fp16:
                print()
                print("   NOTE: Add FP16 support to train_v4_lstm.py:")
                print("   - Use torch.cuda.amp.autocast() for forward pass")
                print("   - Use GradScaler for backward pass")
        else:
            print(f"⚠️  Low VRAM detected. Recommended batch size: {max_safe_batch}")
            print("   Consider:")
            print("   - Gradient accumulation (effective batch = batch_size * accum_steps)")
            print("   - FP16 mixed precision")
            print("   - Reducing model size")
    
    print()
    print("=" * 80)
    print("✅ Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

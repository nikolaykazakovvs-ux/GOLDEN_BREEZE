"""
Benchmark: CPU vs GPU Performance for LSTM Inference and Training
Tests both inference speed and training speed on RTX 3070
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# LSTM Model Architecture
class DirectionLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def benchmark_inference(device, model, data, n_iterations=1000):
    """Benchmark inference speed"""
    model = model.to(device)
    data = data.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(data)
    
    # Sync for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return elapsed, n_iterations / elapsed


def benchmark_training(device, n_epochs=5, batch_size=64, n_samples=10000):
    """Benchmark training speed"""
    # Create synthetic dataset
    X = torch.randn(n_samples, 50, 32)  # seq_len=50, features=32
    y = torch.randint(0, 2, (n_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DirectionLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    model.train()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for epoch in range(n_epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    samples_per_sec = (n_epochs * n_samples) / elapsed
    return elapsed, samples_per_sec


def benchmark_batch_inference(device, model, batch_sizes=[1, 8, 32, 64, 128, 256]):
    """Benchmark inference at different batch sizes"""
    results = {}
    model = model.to(device)
    model.eval()
    
    for bs in batch_sizes:
        data = torch.randn(bs, 50, 32).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        n_iter = max(100, 1000 // bs)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        samples_per_sec = (n_iter * bs) / elapsed
        results[bs] = {
            'time': elapsed,
            'samples_per_sec': samples_per_sec,
            'latency_ms': (elapsed / n_iter) * 1000
        }
    
    return results


def main():
    print("="*70)
    print("BENCHMARK: CPU vs GPU Performance")
    print("="*70)
    
    # System info
    print("\n📊 System Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Cores: {props.multi_processor_count} SMs")
    
    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda') if torch.cuda.is_available() else None
    
    # Create model
    model_cpu = DirectionLSTM()
    model_gpu = DirectionLSTM() if gpu_device else None
    
    # Test data (single sequence - typical for real-time inference)
    single_seq = torch.randn(1, 50, 32)
    
    print("\n" + "="*70)
    print("TEST 1: Single Sequence Inference (Real-time Trading Scenario)")
    print("="*70)
    
    # CPU Inference
    print("\n🔵 CPU Inference (1000 iterations)...")
    cpu_time, cpu_speed = benchmark_inference(cpu_device, model_cpu, single_seq, 1000)
    print(f"   Total Time: {cpu_time:.3f}s")
    print(f"   Speed: {cpu_speed:.1f} predictions/sec")
    print(f"   Latency: {1000/cpu_speed:.3f} ms/prediction")
    
    # GPU Inference
    if gpu_device:
        print("\n🟢 GPU Inference (1000 iterations)...")
        gpu_time, gpu_speed = benchmark_inference(gpu_device, model_gpu, single_seq, 1000)
        print(f"   Total Time: {gpu_time:.3f}s")
        print(f"   Speed: {gpu_speed:.1f} predictions/sec")
        print(f"   Latency: {1000/gpu_speed:.3f} ms/prediction")
        
        speedup = gpu_speed / cpu_speed
        print(f"\n   ⚡ GPU Speedup: {speedup:.2f}x")
    
    print("\n" + "="*70)
    print("TEST 2: Batch Inference (Backtesting Scenario)")
    print("="*70)
    
    batch_sizes = [1, 8, 32, 64, 128, 256, 512]
    
    print("\n🔵 CPU Batch Inference...")
    cpu_batch_results = benchmark_batch_inference(cpu_device, model_cpu, batch_sizes)
    
    if gpu_device:
        print("🟢 GPU Batch Inference...")
        gpu_batch_results = benchmark_batch_inference(gpu_device, model_gpu, batch_sizes)
    
    print("\n📈 Results by Batch Size:")
    print("-"*70)
    print(f"{'Batch':>8} | {'CPU (samples/s)':>15} | {'GPU (samples/s)':>15} | {'Speedup':>10}")
    print("-"*70)
    
    for bs in batch_sizes:
        cpu_sps = cpu_batch_results[bs]['samples_per_sec']
        if gpu_device:
            gpu_sps = gpu_batch_results[bs]['samples_per_sec']
            speedup = gpu_sps / cpu_sps
            print(f"{bs:>8} | {cpu_sps:>15.1f} | {gpu_sps:>15.1f} | {speedup:>9.2f}x")
        else:
            print(f"{bs:>8} | {cpu_sps:>15.1f} | {'N/A':>15} | {'N/A':>10}")
    print("-"*70)
    
    print("\n" + "="*70)
    print("TEST 3: Training Speed (Model Training Scenario)")
    print("="*70)
    
    n_samples = 20000
    n_epochs = 3
    batch_size = 64
    
    print(f"\n📊 Training config: {n_samples} samples, {n_epochs} epochs, batch={batch_size}")
    
    # CPU Training
    print("\n🔵 CPU Training...")
    cpu_train_time, cpu_train_speed = benchmark_training(cpu_device, n_epochs, batch_size, n_samples)
    print(f"   Total Time: {cpu_train_time:.2f}s")
    print(f"   Speed: {cpu_train_speed:.1f} samples/sec")
    print(f"   Time per Epoch: {cpu_train_time/n_epochs:.2f}s")
    
    # GPU Training
    if gpu_device:
        print("\n🟢 GPU Training...")
        gpu_train_time, gpu_train_speed = benchmark_training(gpu_device, n_epochs, batch_size, n_samples)
        print(f"   Total Time: {gpu_train_time:.2f}s")
        print(f"   Speed: {gpu_train_speed:.1f} samples/sec")
        print(f"   Time per Epoch: {gpu_train_time/n_epochs:.2f}s")
        
        train_speedup = gpu_train_speed / cpu_train_speed
        print(f"\n   ⚡ GPU Training Speedup: {train_speedup:.2f}x")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n📋 Performance Comparison:")
    print("-"*70)
    print(f"{'Scenario':<30} | {'CPU':>12} | {'GPU':>12} | {'Speedup':>10}")
    print("-"*70)
    
    print(f"{'Single Inference (pred/s)':<30} | {cpu_speed:>12.1f} | {gpu_speed if gpu_device else 0:>12.1f} | {gpu_speed/cpu_speed if gpu_device else 0:>9.2f}x")
    print(f"{'Batch-256 Inference (samp/s)':<30} | {cpu_batch_results[256]['samples_per_sec']:>12.1f} | {gpu_batch_results[256]['samples_per_sec'] if gpu_device else 0:>12.1f} | {gpu_batch_results[256]['samples_per_sec']/cpu_batch_results[256]['samples_per_sec'] if gpu_device else 0:>9.2f}x")
    print(f"{'Training (samples/s)':<30} | {cpu_train_speed:>12.1f} | {gpu_train_speed if gpu_device else 0:>12.1f} | {gpu_train_speed/cpu_train_speed if gpu_device else 0:>9.2f}x")
    print("-"*70)
    
    print("\n💡 Recommendations:")
    if gpu_device:
        single_speedup = gpu_speed / cpu_speed
        if single_speedup > 1.5:
            print("   ✅ GPU recommended for real-time inference")
        else:
            print("   ⚠️  CPU may be sufficient for single-sequence inference (low latency)")
        
        if train_speedup > 2:
            print(f"   ✅ GPU highly recommended for training ({train_speedup:.1f}x faster)")
        
        print(f"   ✅ GPU essential for batch processing (up to {gpu_batch_results[512]['samples_per_sec']/cpu_batch_results[512]['samples_per_sec']:.1f}x faster)")
    else:
        print("   ⚠️  No GPU available - using CPU only")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    main()

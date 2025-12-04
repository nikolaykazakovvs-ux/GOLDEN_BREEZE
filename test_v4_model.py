"""
Quick test for GoldenBreezeFusionV4 model.
Run: python test_v4_model.py
"""

import torch
from aimodule.models.v4_transformer import GoldenBreezeFusionV4, V4Config

def main():
    print("=" * 60)
    print("GoldenBreezeFusionV4 - Quick Test")
    print("=" * 60)
    
    # Create model
    config = V4Config()
    model = GoldenBreezeFusionV4(config)
    
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  nhead: {config.nhead}")
    print(f"  num_layers_fast: {config.num_layers_fast}")
    print(f"  num_layers_slow: {config.num_layers_slow}")
    print(f"  patch_size: {config.patch_size}")
    print(f"  patch_stride: {config.patch_stride}")
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nParameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    
    inputs = {
        "x_fast_ohlcv": torch.randn(4, config.seq_len_fast, config.input_channels),
        "x_slow_ohlcv": torch.randn(4, config.seq_len_slow, config.input_channels),
        "smc_static": torch.randn(4, config.static_smc_dim),
        "smc_dynamic": torch.randn(4, config.max_dynamic_tokens, config.dynamic_smc_dim),
    }
    
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    
    print(f"\nOutputs:")
    print(f"  class_logits: {output['class_logits'].shape}")
    print(f"  class_probs: {output['class_probs'].shape}")
    print(f"  score: {output['score'].shape}")
    print(f"  gate_value: {output['gate_value']:.4f}")
    
    # Test prediction
    print(f"\nTesting prediction...")
    pred = model.predict(**{k: v[:1] for k, v in inputs.items()})
    print(f"  direction: {pred['direction']}")
    print(f"  confidence: {pred['confidence']:.4f}")
    print(f"  score: {pred['score']:.4f}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print(f"\nTesting on GPU ({torch.cuda.get_device_name(0)})...")
        model_gpu = model.cuda()
        inputs_gpu = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            output_gpu = model_gpu(**inputs_gpu)
        
        print(f"  GPU forward pass: OK")
        print(f"  class_logits device: {output_gpu['class_logits'].device}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

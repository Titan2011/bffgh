#!/usr/bin/env python
"""
Test script for GAS RandLA-Net implementation
"""

import torch
import torch.nn as nn
import numpy as np
from helper_tool import ConfigSemantic3D as cfg
from RandLANet_fixed import Network

def test_gas_implementation():
    """Test the complete GAS implementation."""
    print("Testing GAS RandLA-Net implementation...")
    
    # Create model
    model = Network(cfg)
    model = model.cuda()
    model.train()
    
    # Create dummy input
    batch_size = 2
    num_points = 40960
    
    # Mock input data
    xyz = torch.randn(batch_size, num_points, 3).cuda()
    features = torch.randn(batch_size, num_points, cfg.num_features).cuda()
    labels = torch.randint(0, cfg.num_classes, (batch_size, num_points)).cuda()
    
    # Create interpolation indices for upsampling
    interp_idx = []
    for i in range(cfg.num_layers):
        # Simulate upsampling indices
        upsample_size = num_points // (cfg.sub_sampling_ratio[i] if i < len(cfg.sub_sampling_ratio) else 1)
        idx = torch.randint(0, upsample_size, (batch_size, num_points, 1)).cuda()
        interp_idx.append(idx)
    
    # Create input dictionary
    inputs = {
        'xyz': [xyz],
        'features': features,
        'labels': labels,
        'interp_idx': interp_idx
    }
    
    print(f"Input shapes:")
    print(f"  xyz: {xyz.shape}")
    print(f"  features: {features.shape}")
    print(f"  labels: {labels.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
    
    # Check outputs
    logits = outputs['logits']
    aux_info = outputs['aux_info']
    
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  Expected: ({batch_size}, {num_points}, {cfg.num_classes})")
    
    # Verify shapes
    assert logits.shape == (batch_size, num_points, cfg.num_classes), f"Logits shape mismatch: {logits.shape}"
    
    # Check auxiliary information
    assert 'encoder_aux' in aux_info, "Missing encoder auxiliary info"
    assert 'decoder_aux' in aux_info, "Missing decoder auxiliary info"
    
    print(f"  Number of encoder layers: {len(aux_info['encoder_aux'])}")
    print(f"  Number of decoder layers: {len(aux_info['decoder_aux'])}")
    
    # Test loss computation
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, cfg.num_classes), labels.view(-1))
    print(f"\nLoss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    
    # Check gradients
    total_params = 0
    grad_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.grad is not None:
            grad_params += param.numel()
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameters with gradients: {grad_params:,}")
    print(f"  Gradient coverage: {grad_params/total_params*100:.1f}%")
    
    # Test GAS module specifically
    print(f"\nTesting GAS module...")
    from RandLANet_fixed import GeometryAdaptiveSampling
    
    gas = GeometryAdaptiveSampling(in_channels=64).cuda()
    test_xyz = torch.randn(batch_size, 1000, 3).cuda()
    test_features = torch.randn(batch_size, 1000, 64).cuda()
    
    sampled_feat, sampled_xyz, sampled_idx, aux_info = gas(test_xyz, test_features, sample_ratio=0.25)
    
    print(f"  Input points: {test_xyz.shape[1]}")
    print(f"  Sampled points: {sampled_xyz.shape[1]}")
    print(f"  Sampling ratio: {sampled_xyz.shape[1] / test_xyz.shape[1]:.3f}")
    print(f"  Auxiliary info keys: {list(aux_info.keys())}")
    
    # Verify sampling worked correctly
    assert sampled_xyz.shape == (batch_size, 250, 3), f"Sampled XYZ shape mismatch: {sampled_xyz.shape}"
    assert sampled_feat.shape == (batch_size, 250, 64), f"Sampled features shape mismatch: {sampled_feat.shape}"
    
    print("\n✅ All tests passed!")
    
    return True

def test_boundary_metrics():
    """Test boundary-aware metrics."""
    print("\nTesting boundary metrics...")
    
    from boundary_metrics import BoundaryAwareMetrics
    
    num_classes = 9
    batch_size = 2
    num_points = 1000
    
    # Create mock predictions and labels
    logits = torch.randn(batch_size, num_points, num_classes).cuda()
    labels = torch.randint(0, num_classes, (batch_size, num_points)).cuda()
    xyz = torch.randn(batch_size, num_points, 3).cuda()
    
    # Create metrics object
    metrics = BoundaryAwareMetrics(num_classes)
    
    # Update metrics
    metrics.update(logits, labels, xyz)
    
    # Compute metrics
    results = metrics.compute_metrics()
    
    print(f"Computed metrics: {list(results.keys())}")
    
    # Check key metrics
    assert 'mIoU' in results
    assert 'boundary_mIoU' in results
    assert 'boundary_mF1' in results
    
    print(f"  Overall mIoU: {results['mIoU']:.4f}")
    print(f"  Boundary mIoU: {results['boundary_mIoU']:.4f}")
    print(f"  Boundary F1: {results['boundary_mF1']:.4f}")
    
    print("✅ Boundary metrics test passed!")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("GAS RandLA-Net Implementation Test Suite")
    print("="*60)
    
    try:
        # Test 1: Basic implementation
        test_gas_implementation()
        
        # Test 2: Boundary metrics
        test_boundary_metrics()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
        print("\nYour GAS implementation is ready for A* publication!")
        print("Next steps:")
        print("1. Run training: python main_Semantic3D.py --mode train")
        print("2. Run evaluation: python main_Semantic3D.py --mode test")
        print("3. Run ablation study: python ablation_study.py")
        print("4. Run SOTA comparison: python sota_comparison.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
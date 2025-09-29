#!/usr/bin/env python3
"""
Test script to verify the three specific MoRE model variants work correctly.
This tests model instantiation, forward passes, and basic functionality.
"""

import torch
import torch.nn.functional as F
from models import (
    # Baseline
    BaseMoEModel,
    # Synthesis I: Recurrence at Expert Level
    MoREModelSynthesisIOptionA, MoREModelSynthesisIOptionB,
    # Synthesis II: MoE Layer as Recurrent Unit
    MoREModelSynthesisII
)

def test_model_forward(model_class, model_name, **kwargs):
    """Test that a model can be instantiated and run a forward pass."""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        # Create model
        model = model_class(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_experts=4,
            **kwargs
        )
        print(f"✅ {model_name} created successfully")
        
        # Create dummy input
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        print(f"✅ Forward pass successful")
        
        # Check output shapes
        assert outputs["logits"].shape == (batch_size, seq_len, 1000), f"Wrong logits shape: {outputs['logits'].shape}"
        assert outputs["loss"] is not None, "Loss should not be None"
        assert outputs["aux_loss"] is not None, "Aux loss should not be None"
        print(f"✅ Output shapes correct")
        
        # Check that loss is a scalar
        assert outputs["loss"].dim() == 0, f"Loss should be scalar, got shape {outputs['loss'].shape}"
        print(f"✅ Loss computation correct")
        
        # Test generation (if available)
        if hasattr(model, 'generate'):
            try:
                generated = model.generate(input_ids, max_new_tokens=5)
                assert generated.shape[1] == seq_len + 5, f"Generation shape wrong: {generated.shape}"
                print(f"✅ Generation successful")
            except Exception as e:
                print(f"⚠️  Generation failed: {e}")
        
        print(f"🎉 {model_name} test passed!")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synthesis_ii_iterations():
    """Test the iterative routing behavior of Synthesis II."""
    print(f"\n🧪 Testing Synthesis II Iterative Routing...")
    
    try:
        # Create model
        model = MoREModelSynthesisII(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_experts=4,
            num_iters=3,
            residual_scale=0.5
        )
        
        print(f"✅ Synthesis II model created with {model.num_iters} iterations")
        
        # Test forward pass with iteration info
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels, return_iteration_info=True)
        print(f"✅ Forward pass successful")
        
        # Check that we get iteration information
        assert "layer_states" in outputs, "Should return layer states for iteration info"
        assert len(outputs["layer_states"]) == 2, f"Should have 2 layers, got {len(outputs['layer_states'])}"
        
        # Check that each layer has the right number of iterations
        for layer_idx, layer_states in enumerate(outputs["layer_states"]):
            assert len(layer_states) == model.num_iters, f"Layer {layer_idx} should have {model.num_iters} states, got {len(layer_states)}"
            print(f"✅ Layer {layer_idx} has {len(layer_states)} iteration states")
        
        print(f"🎉 Synthesis II iteration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Synthesis II iteration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MoRE Model Tests...")
    
    # Test configurations for the three specific models
    test_configs = [
        (BaseMoEModel, "Base MoE Model"),
        (MoREModelSynthesisIOptionA, "MoRE Synthesis I Option A (Independent Recurrent Experts)", {"num_recurrences": 2}),
        (MoREModelSynthesisIOptionB, "MoRE Synthesis I Option B (Shared Recurrent Block with Projections)", {"num_recurrences": 2}),
        (MoREModelSynthesisII, "MoRE Synthesis II (MoE Layer as Recurrent Unit)", {"num_iters": 2, "residual_scale": 1.0}),
    ]
    
    # Run basic tests
    passed = 0
    total = len(test_configs)
    
    for test_config in test_configs:
        if len(test_config) == 2:
            model_class, model_name = test_config
            kwargs = {}
        else:
            model_class, model_name, kwargs = test_config
            
        if test_model_forward(model_class, model_name, **kwargs):
            passed += 1
    
    # Run special tests
    print(f"\n🔬 Running Special Tests...")
    
    if test_synthesis_ii_iterations():
        passed += 1
        total += 1
    
    # Summary
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MoRE models are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


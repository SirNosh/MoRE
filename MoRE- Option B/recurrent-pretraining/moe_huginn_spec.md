# Shared Recurrent Block with Projections Implementation Specification

## Overview

This specification details how to implement the **Shared Recurrent Block with Projections** architecture by modifying the Huginn codebase from the `recurrent-pretraining` repository. The architecture combines Mixture-of-Experts (MoE) routing with the proven recurrent depth approach of Huginn.

## Architecture Summary

```
Input Tokens → MoE Router → Expert Selection → N Lightweight FFN Experts → Shared Recurrent Block → Output
```

**Key Components:**
- **MoE Router**: Selects k experts per token
- **N Lightweight FFN Experts**: Simple 1-layer projections for shallow specialization and 1 shared expert
- **Shared Recurrent Block**: The core Huginn recurrent reasoning engine
- **Parameter Efficiency**: Only one shared recurrent block + lightweight expert projections

## 1. Repository Setup and Environment

### 1.1 Clone and Setup
```bash
git clone https://github.com/seal-rg/recurrent-pretraining.git
cd recurrent-pretraining
```

### 1.2 Environment Configuration
Follow the existing setup in the repository:
- Use the same Python environment and dependencies
- Ensure PyTorch compatibility with the existing codebase
- Install additional dependencies if needed for MoE implementation

### 1.3 Dataset Preparation
Use the existing Huginn dataset:
```bash
# The dataset is already available at:
# https://huggingface.co/datasets/tomg-group-umd/huginn-dataset
```

## 2. Core Implementation Changes

### 2.1 New Files to Create

#### 2.1.1 `recpre/moe_layers.py`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MoERouter(nn.Module):
    """Top-k MoE router for expert selection"""
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            routing_weights: [batch_size, seq_len, k] - normalized weights
            selected_experts: [batch_size, seq_len, k] - expert indices
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute router logits
        router_logits = self.gate(x)  # [batch, seq, num_experts]
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_logits, self.k, dim=-1
        )
        
        # Normalize weights (softmax over selected experts)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights, selected_experts

class LightweightFFNExpert(nn.Module):
    """Lightweight 1-layer FFN expert"""
    def __init__(self, d_model: int, expansion_factor: float = 2.0):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        
        self.input_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.output_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            transformed_x: [batch_size, seq_len, d_model]
        """
        h = self.input_proj(x)
        h = self.activation(h)
        h = self.output_proj(h)
        return self.dropout(h)

class MoELayer(nn.Module):
    """MoE layer with lightweight experts"""
    def __init__(self, d_model: int, num_experts: int, k: int = 2, 
                 expansion_factor: float = 2.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        # Router
        self.router = MoERouter(d_model, num_experts, k)
        
        # Experts
        self.experts = nn.ModuleList([
            LightweightFFNExpert(d_model, expansion_factor)
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        routing_weights, selected_experts = self.router(x)  # [B, S, k], [B, S, k]
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token through selected experts
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_input = x[batch_idx, seq_idx:seq_idx+1, :]  # [1, 1, d_model]
                token_output = torch.zeros_like(token_input)
                
                for k_idx in range(self.k):
                    expert_idx = selected_experts[batch_idx, seq_idx, k_idx].item()
                    weight = routing_weights[batch_idx, seq_idx, k_idx]
                    
                    expert_output = self.experts[expert_idx](token_input)
                    token_output += weight * expert_output
                
                output[batch_idx, seq_idx:seq_idx+1, :] = token_output
        
        return output
```

#### 2.1.2 `recpre/moe_model_dynamic.py`
Modify the existing `model_dynamic.py` to include MoE layers:

```python
# Import the existing Huginn model components
from .model_dynamic import *  # Import existing model components
from .moe_layers import MoELayer

class MoERecurrentBlock(nn.Module):
    """MoE + Shared Recurrent Block combination"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = getattr(config, 'num_experts', 8)
        self.k = getattr(config, 'moe_k', 2)
        
        # MoE layer for lightweight expert specialization
        self.moe_layer = MoELayer(
            d_model=self.d_model,
            num_experts=self.num_experts,
            k=self.k,
            expansion_factor=getattr(config, 'moe_expansion_factor', 2.0)
        )
        
        # Shared recurrent block (using existing Huginn block)
        self.shared_recurrent = RecurrentBlock(config)  # Use existing implementation
        
        # Layer norm after MoE
        self.norm = nn.RMSNorm(self.d_model, eps=config.norm_eps)
        
    def forward(self, x: torch.Tensor, latent_state: torch.Tensor, 
                num_recurrence: int) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            latent_state: Initial latent state [batch_size, seq_len, d_model]
            num_recurrence: Number of recurrent iterations
        Returns:
            final_state: Output after recurrence [batch_size, seq_len, d_model]
        """
        # Step 1: Apply MoE layer for expert specialization
        expert_modified_x = self.moe_layer(x)
        expert_modified_x = self.norm(expert_modified_x)
        
        # Step 2: Apply shared recurrent block for r iterations
        current_state = latent_state
        for step in range(num_recurrence):
            current_state = self.shared_recurrent(
                expert_modified_x, 
                current_state
            )
        
        return current_state

class MoEHuginnModel(nn.Module):
    """Modified Huginn model with MoE integration"""
    def __init__(self, config):
        super().__init__()
        # Copy existing Huginn architecture setup
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Prelude (input embedding layers) - use existing implementation
        self.prelude = PreludeBlock(config)
        
        # Core: MoE + Shared Recurrent Block
        self.core = MoERecurrentBlock(config)
        
        # Coda (output layers) - use existing implementation  
        self.coda = CodaBlock(config)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # Use existing Huginn initialization strategy
        # Copy from the original model_dynamic.py
        pass
    
    def forward(self, input_ids: torch.Tensor, num_recurrence: Optional[int] = None):
        """Forward pass through MoE-Huginn model"""
        if num_recurrence is None:
            num_recurrence = getattr(self.config, 'mean_recurrence', 16)
        
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Prelude: embed into latent space
        latent_x = self.prelude(x)
        
        # Initialize random latent state (following Huginn approach)
        latent_state = torch.randn(
            batch_size, seq_len, self.d_model,
            device=input_ids.device,
            dtype=latent_x.dtype
        ) * self.config.init_std
        
        # Core: MoE + Shared Recurrent Block
        final_state = self.core(latent_x, latent_state, num_recurrence)
        
        # Coda: decode to vocabulary
        logits = self.coda(final_state)
        
        return logits
```

### 2.2 Configuration Changes

#### 2.2.1 Update `recpre/model_registry.py`
Add the new MoE-Huginn model configuration:

```python
# Add to existing configurations
MOE_HUGINN_CONFIGS = {
    "moe-huginn-3.5b": {
        # Base Huginn config
        "d_model": 2560,
        "n_heads": 20, 
        "n_layers_prelude": 4,
        "n_layers_core": 1,
        "n_layers_coda": 4,
        "vocab_size": 131072,
        "mean_recurrence": 16,
        
        # MoE specific parameters
        "num_experts": 8,
        "moe_k": 2,
        "moe_expansion_factor": 2.0,
        
        # Other parameters from original Huginn
        "max_seq_len": 4096,
        "norm_eps": 1e-5,
        "init_std": 0.02,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }
}

# Update the registry
MODEL_CONFIGS.update(MOE_HUGINN_CONFIGS)
```

### 2.3 Training Script Modifications

#### 2.3.1 Update `train.py`
Modify the main training script to support the new architecture:

```python
# Add import for the new model
from recpre.moe_model_dynamic import MoEHuginnModel

def create_model(config):
    """Create model based on configuration"""
    if config.model_type == "moe_huginn":
        return MoEHuginnModel(config)
    else:
        # Use original model creation
        return create_original_model(config)

# Ensure training loop handles MoE-specific aspects
def training_step(model, batch, config):
    """Modified training step for MoE model"""
    input_ids = batch['input_ids']
    
    # Sample recurrence steps (keep existing Huginn approach)
    if hasattr(config, 'recurrence_distribution'):
        num_recurrence = sample_recurrence_steps(config)
    else:
        num_recurrence = config.mean_recurrence
    
    # Forward pass
    logits = model(input_ids, num_recurrence=num_recurrence)
    
    # Compute loss (same as original)
    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1].contiguous()
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    return loss
```

## 3. Implementation Strategy

### 3.1 Phase 1: Basic Implementation
1. **Create MoE components** (`moe_layers.py`)
2. **Integrate with existing Huginn blocks** (`moe_model_dynamic.py`)
3. **Update configuration system** 
4. **Modify training script minimally**

### 3.2 Phase 2: Training Infrastructure
1. **Adapt data loading** (use existing Huginn dataset)
2. **Update evaluation scripts** 
3. **Ensure compatibility with existing benchmarking**
4. **Add MoE-specific metrics** (expert utilization, routing entropy)

### 3.3 Phase 3: Optimization and Scaling
1. **Optimize MoE routing efficiency**
2. **Add load balancing loss** for expert utilization
3. **Implement gradient checkpointing** for MoE layers
4. **Scale to full dataset**

## 4. Training Configuration

### 4.1 Model Size Comparison
- **Original Huginn-3.5B**: 3.5B parameters
- **MoE-Huginn**: ~3.5B + (N × lightweight_expert_params)
  - With 8 experts, 2x expansion: ~3.8B parameters
  - Still much smaller than traditional 50B models with equivalent compute

### 4.2 Training Hyperparameters
```yaml
# config/moe_huginn_config.yaml
model_type: "moe_huginn"
model_name: "moe-huginn-3.5b"

# MoE specific
num_experts: 8
moe_k: 2
moe_expansion_factor: 2.0
load_balancing_loss_weight: 0.01

# Keep existing Huginn parameters
mean_recurrence: 16
truncated_backprop_steps: 8
batch_size: 16384  # tokens per step
learning_rate: 1e-4
warmup_steps: 2000

# Training data
dataset: "tomg-group-umd/huginn-dataset"
max_tokens: 800_000_000_000  # 800B tokens like original
```

### 4.3 Evaluation Strategy
1. **Compare against original Huginn** on same benchmarks
2. **Measure expert utilization** and routing patterns
3. **Test compute scaling** (1-50 recurrence steps)
4. **Benchmark on reasoning tasks** (GSM8K, MATH, etc.)

## 5. Expected Benefits

### 5.1 Architecture Advantages
- **Minimal parameter overhead**: Only lightweight FFN experts added
- **Shared recurrent reasoning**: Leverages proven Huginn approach  
- **Expert specialization**: Different experts for different token types
- **Scalable**: Can increase experts without changing core architecture

### 5.2 Performance Expectations
- **Better token-level adaptation**: Experts can specialize per token type
- **Maintained recurrent benefits**: All recurrent depth advantages preserved
- **Improved efficiency**: Router can select most relevant processing
- **Enhanced reasoning**: Combination of specialization + recurrence

## 6. Implementation Checklist

### 6.1 Core Components
- [ ] Implement `MoERouter` class
- [ ] Implement `LightweightFFNExpert` class  
- [ ] Implement `MoELayer` class
- [ ] Create `MoERecurrentBlock` combining MoE + Huginn
- [ ] Integrate with existing `PreludeBlock` and `CodaBlock`

### 6.2 Training Infrastructure
- [ ] Update model registry with MoE configs
- [ ] Modify training loop for MoE
- [ ] Add expert utilization logging
- [ ] Update evaluation scripts
- [ ] Test on small scale first

### 6.3 Validation
- [ ] Verify parameter count matches expectations
- [ ] Test forward/backward pass
- [ ] Validate expert routing behavior  
- [ ] Compare against baseline Huginn
- [ ] Scale up gradually

## 7. Monitoring and Debugging

### 7.1 Key Metrics to Track
- **Expert utilization balance**: Ensure all experts are used
- **Routing entropy**: Measure routing diversity
- **Loss convergence**: Compare to baseline Huginn
- **Memory usage**: Monitor additional overhead
- **Training stability**: Check for routing collapse

### 7.2 Common Issues and Solutions
- **Expert imbalance**: Add load balancing loss
- **Routing collapse**: Increase routing temperature
- **Memory issues**: Use gradient checkpointing
- **Training instability**: Adjust initialization schemes

This specification provides a complete roadmap for implementing the Shared Recurrent Block with Projections architecture using the existing Huginn codebase as the foundation.
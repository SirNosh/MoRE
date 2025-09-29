
# MoRE: Mixture of Recurrent Experts - Comprehensive Research Guide

## Table of Contents
1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Detailed Architecture Descriptions](#2-detailed-architecture-descriptions)
3. [Implementation Specifications and Pipeline](#3-implementation-specifications-and-pipeline)
4. [Datasets and Evaluation Methodology](#4-datasets-and-evaluation-methodology)
5. [Architecture Validation Metrics](#5-architecture-validation-metrics)
6. [Technology Stack and Tools](#6-technology-stack-and-tools)
7. [Technical Implementation Details](#7-technical-implementation-details)

---

## 1. Introduction and Motivation

### 1.1 Background: The Evolution of Large Language Models

Large Language Models (LLMs) have revolutionized natural language processing through the Transformer architecture [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). However, as models scale to billions of parameters, computational costs become prohibitive. Two key innovations have emerged to address this challenge:

1. **Mixture of Experts (MoE)**: Introduced by [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538) in "Outrageously Large Neural Networks," MoE enables training models with trillions of parameters by activating only a subset of the network for each input token. The Switch Transformer [Fedus et al., 2021](https://arxiv.org/abs/2101.03961) simplified this approach with single-expert routing, demonstrating that sparse models can achieve better performance per FLOP than dense models.

2. **Recurrent Processing in Transformers**: Recent work has shown that adding recurrence to Transformers can improve reasoning capabilities. The Recurrent Memory Transformer (RMT) [Bulatov et al., 2022](https://arxiv.org/abs/2207.06881) extends context length through recurrent memory tokens. Most recently, the "Scaling Test-Time Compute with Latent Reasoning" paper [Geiping et al., 2025](https://arxiv.org/abs/2502.05171) demonstrated that iterative processing in latent space dramatically improves reasoning without generating explicit reasoning tokens.

### 1.2 The MoRE Concept: Bridging MoE and Recurrence

**Mixture of Recurrent Experts (MoRE)** combines these two paradigms by replacing the standard feed-forward network (FFN) experts in MoE with recurrent processing blocks. This fusion aims to achieve:

- **Parameter Efficiency**: Like MoE, only a subset of experts activate per token
- **Computational Depth**: Like latent reasoning models, experts can perform iterative computation
- **Reasoning Capability**: Recurrent experts can potentially solve multi-step problems within a single forward pass

The key insight is that while standard MoE increases model capacity through more parameters, MoRE increases effective computational depth through recurrence, potentially achieving similar or better performance with fewer total parameters.

### 1.3 Research Questions

This research investigates:
1. Can recurrent experts outperform standard FFN experts in reasoning tasks?
2. What is the optimal way to integrate recurrence into MoE architectures?
3. How do different recurrent expert designs trade off between computational cost and model quality?
4. Can MoRE models achieve better parameter efficiency than standard MoE?

### 1.4 Related Work

- **MoE in Language Models**: GShard [Lepikhin et al., 2020](https://arxiv.org/abs/2006.16668) scaled MoE to 600B parameters. GLaM [Du et al., 2021](https://arxiv.org/abs/2112.06905) achieved 1.2T parameters. ST-MoE [Zoph et al., 2022](https://arxiv.org/abs/2202.08906) improved routing stability.

- **Recurrence in Transformers**: Universal Transformers [Dehghani et al., 2018](https://arxiv.org/abs/1807.03819) applied weight sharing across layers. ALBERT [Lan et al., 2019](https://arxiv.org/abs/1909.11942) demonstrated cross-layer parameter sharing. [Jain et al., 2023](https://arxiv.org/abs/2306.03214) replaced FFNs with BiGRUs in Q-learning.

- **Iterative Reasoning**: Chain-of-thought prompting [Wei et al., 2022](https://arxiv.org/abs/2201.11903) showed explicit reasoning helps. The Huginn model [Geiping et al., 2025] demonstrated that latent reasoning can achieve similar gains without token generation overhead.

---

## 2. Detailed Architecture Descriptions

### 2.1 Standard MoE Baseline

Before describing the MoRE variants, we establish the baseline Mixture of Experts architecture:

```python
class StandardMoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=1):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits.softmax(dim=-1), self.top_k
        )
        
        # Route tokens to selected experts
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = expert(expert_input)
                output[expert_mask] += expert_output * routing_weights[expert_mask, :, i:i+1]
                
        return output
```

Key characteristics:
- Each expert is a standard 2-layer FFN: Linear → Activation → Linear
- Routing is performed via learned softmax over expert scores
- Top-k routing selects k experts per token (typically k=1 or k=2)
- Load balancing loss ensures all experts are utilized

### 2.2 Synthesis I: Recurrence at the Expert Level

#### 2.2.1 Recurrent Experts (Deep Specialization)

```python
class RecurrentExpert(nn.Module):
    """Huginn-style recurrent expert with weight tying across r steps."""
    def __init__(self, d_model: int, num_steps: int = 2, n_heads: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.adapter = nn.Linear(2 * d_model, d_model)
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads=n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, 4 * d_model)
        self.norm3 = RMSNorm(d_model)
        self.norm4 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, return_all_steps: bool = False):
        state = torch.randn_like(x) * 0.02
        static = x
        states = []
        for _ in range(self.num_steps):
            state = self.adapter(torch.cat([state, static], dim=-1))
            a_out, _ = self.attn(self.norm1(state))
            state = self.norm2(a_out + state)
            m_out = self.mlp(self.norm3(state))
            state = self.norm4(m_out + state)
            states.append(state)
        return states if return_all_steps else state
```

Key points:
- Each expert owns its own recurrent block parameters (no sharing).
- Compute ∝ `k × r`; parameters ∝ `N × |R|`.
- Best for maximal specialization but highest memory/FLOPs.

#### 2.2.2 Shared Recurrent Block within Experts (Shallow Specialization)

```python
class SharedRecurrentEngine(nn.Module):
    def __init__(self, d_model: int, num_steps: int = 2, n_heads: int = 8):
        super().__init__()
        self.core = RecurrentExpert(d_model, num_steps=num_steps, n_heads=n_heads)

    def forward(self, u: torch.Tensor):
        return self.core(u)

class ProjectionWrappedExpert(nn.Module):
    """Expert i with unique projections around shared engine R."""
    def __init__(self, d_model: int, shared_engine: SharedRecurrentEngine):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.R = shared_engine  # shared among experts in this layer

    def forward(self, x: torch.Tensor):
        u = self.W_in(x)
        v = self.R(u)
        y = self.W_out(v)
        return y
```

Key points:
- One recurrent engine `R` per MoE layer; experts differ by `{W_in^i, W_out^i}` only.
- Compute ∝ `k × r`; parameters dominated by `|R| + 2N × d_model^2`.
- Natural for continual learning: add a new expert by training a new projection pair + router.

Option A completely replaces each FFN expert with a recurrent block based on the Huginn architecture:

```python
class RecurrentExpert(nn.Module):
    """Based on Huginn's latent reasoning block"""
    def __init__(self, d_model, num_steps=1):
        super().__init__()
        # Input adapter merges current state with static embedding
        self.adapter = nn.Linear(2 * d_model, d_model)
        
        # Core recurrent block (simplified from Huginn)
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads=8)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, d_model * 4)  # SwiGLU style
        self.norm3 = RMSNorm(d_model)
        self.norm4 = RMSNorm(d_model)
        
        self.num_steps = num_steps
        
    def forward(self, x, return_all_steps=False):
        # Initialize latent state (could be random as in Huginn)
        latent_state = torch.randn_like(x) * 0.02
        static_input = x.clone()
        
        states = []
        for step in range(self.num_steps):
            # Adapt: merge latent state with original input
            merged = torch.cat([latent_state, static_input], dim=-1)
            latent_state = self.adapter(merged)
            
            # Self-attention (can attend to previous states if cached)
            attn_out, _ = self.attn(self.norm1(latent_state))
            latent_state = self.norm2(attn_out + latent_state)
            
            # Gated MLP
            mlp_out = self.mlp(self.norm3(latent_state))
            latent_state = self.norm4(mlp_out + latent_state)
            
            states.append(latent_state)
            
        return states if return_all_steps else latent_state
```

**Key Design Choices:**
- Replaces static 2-layer FFN with iterative transformer block
- Can run for variable steps (more steps = more computation = potentially better reasoning)
- Includes self-attention within expert (tokens routed to same expert can interact)
- Uses Huginn's adapter mechanism to inject original input at each step
- Parameters are reused across steps (weight tying)

**Advantages:**
- Can scale computation at test time by increasing steps
- Parameter efficient (small block reused many times)
- Proven architecture from Huginn achieving strong reasoning results

**Challenges:**
- Sequential computation reduces parallelism
- May require careful initialization and training procedures
- Gradient flow through many steps needs management

### 2.3 Synthesis II: Recurrence at the MoE Layer Level

#### 2.3.1 MoE Layer as the Recurrent Unit

```python
class RecurrentMoELayer(nn.Module):
    """Treat the entire MoE (router + experts) as a recurrent block."""
    def __init__(self, moe_layer: nn.Module, num_iters: int = 2, residual_scale: float = 1.0):
        super().__init__()
        self.moe = moe_layer
        self.num_iters = num_iters
        self.residual_scale = residual_scale

    def forward(self, s: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        state = s
        all_states = []
        for _ in range(self.num_iters):
            delta, _router = self.moe(state)  # standard MoE forward
            state = state + self.residual_scale * delta
            all_states.append(state)
        return state, all_states
```

Router behavior across iterations:
- Router observes `state_t`; selections adapt over iterations, enabling latent chain-of-thought.
- Stabilization: residual scaling, entropy floors on routing, EMA on logits, truncated BPTT.

Option B maintains FFN structure but inserts recurrence in the middle:

```python
class SandwichRecurrentExpert(nn.Module):
    def __init__(self, d_model, d_ff, hidden_steps=2):
        super().__init__()
        # Expand to higher dimension
        self.expand = nn.Linear(d_model, d_ff)
        
        # Recurrent processing in expanded space
        self.recurrent_cell = nn.GRUCell(d_ff, d_ff)
        self.hidden_steps = hidden_steps
        
        # Contract back to model dimension
        self.contract = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # x shape: [batch, d_model]
        
        # Expand
        h = self.expand(x)  # [batch, d_ff]
        h = F.gelu(h)  # Activation after expansion
        
        # Recurrent refinement in expanded space
        for _ in range(self.hidden_steps):
            h = self.recurrent_cell(h, h)  # GRU step
            
        # Contract
        output = self.contract(h)  # [batch, d_model]
        return output
```

**Key Design Choices:**
- Preserves FFN's expansion-contraction pattern
- Recurrence operates in high-dimensional space (typically 4x model dim)
- Can be seen as iterative refinement of features
- Maintains similar interface to standard FFN

**Advantages:**
- Natural integration with existing transformer code
- High-dimensional recurrence may capture complex patterns
- Can initialize from pretrained FFN weights

**Challenges:**
- Higher memory usage due to large hidden states
- Recurrence in high dimensions may be unstable
- More parameters than Option A

### 2.4 MoRE Option C: Parallel FFN and Recurrent Branches

Option C runs FFN and recurrent paths in parallel, combining outputs:

```python
class ParallelRecurrentExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # FFN branch
        self.ffn = FeedForward(d_model, d_ff)
        
        # Recurrent branch
        self.rnn = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        
        # Gating mechanism to combine branches
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 branches
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        
        # FFN branch (position-wise)
        ffn_out = self.ffn(x)
        
        # RNN branch (sequential)
        rnn_out, _ = self.rnn(x)
        
        # Compute gates
        gates = self.gate_net(x)  # [batch, seq_len, 2]
        
        # Weighted combination
        output = gates[..., 0:1] * ffn_out + gates[..., 1:2] * rnn_out
        
        return output
```

**Key Design Choices:**
- Two specialized pathways: FFN for position-wise patterns, RNN for sequential
- Learned gating allows model to choose pathway per token
- Can degenerate to pure FFN or pure RNN if beneficial
- Parallel computation possible (though limited on single GPU)

**Advantages:**
- Best of both worlds: FFN efficiency and RNN sequence modeling
- Graceful degradation if one branch underperforms
- Gating provides interpretability

**Challenges:**
- Nearly doubles parameters and computation
- Gating may collapse to favor one branch
- Training dynamics between branches need balancing

### 2.5 MoRE Option D: Sequential FFN-RNN Processing

Option D applies recurrence as a refinement step after FFN:

```python
class SequentialRecurrentExpert(nn.Module):
    def __init__(self, d_model, d_ff, refine_steps=1):
        super().__init__()
        # Standard FFN first
        self.ffn = FeedForward(d_model, d_ff)
        
        # Recurrent refinement
        self.refine_cell = nn.LSTMCell(d_model, d_model)
        self.refine_steps = refine_steps
        
        # Residual gate for refinement
        self.refine_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial FFN processing
        ffn_out = self.ffn(x)
        
        # Iterative refinement
        h = ffn_out
        c = torch.zeros_like(h)  # LSTM cell state
        
        for _ in range(self.refine_steps):
            h_new, c = self.refine_cell(h, (h, c))
            
            # Gated residual connection
            gate = self.refine_gate(h)
            h = gate * h_new + (1 - gate) * h
            
        return h
```

**Key Design Choices:**
- FFN provides initial transformation
- RNN refines the FFN output iteratively
- Residual connections ensure stability
- Can be viewed as error correction on FFN output

**Advantages:**
- Minimal overhead (small RNN addition)
- Easy to retrofit to existing models
- Clear separation of concerns

**Challenges:**
- Limited improvement potential
- RNN may learn to be identity if FFN is sufficient
- Less radical than other options

---

## 3. Implementation Specifications and Pipeline

### 3.1 Model Specifications

All models share a common backbone with variations in the expert implementation:

**Common Architecture:**
```yaml
Base Configuration:
  model_type: decoder-only transformer
  num_layers: 12
  d_model: 512
  num_heads: 8
  head_dim: 64
  vocab_size: 32000
  max_seq_length: 512
  activation: gelu
  
MoE Configuration:
  num_experts: 8
  top_k: 1  # Switch-style routing
  capacity_factor: 1.0
  aux_loss_weight: 0.001
  router_z_loss_weight: 0.001
  expert_dropout: 0.0  # Can be increased for regularization
```

### 3.2 Detailed Specifications per Synthesis

#### Synthesis I — Recurrent Experts (Deep Specialization)
```yaml
Expert Configuration:
  type: per_expert_recurrent
  hidden_size: 512
  num_recurrent_steps:
    train: 2-4 (randomized)
    inference: up to 16
  attention_heads: 8
  mlp_ratio: 4.0
  use_gated_mlp: true
  adapter_type: linear
  initialization:
    latent_state: random_normal(std=0.02)
    adapter: xavier_uniform

Estimates:
  total_params: high (∝ N × |R|)
  active_params_per_token: high
  throughput: lower with larger r
```

#### Synthesis I — Shared Recurrent Block within Experts (Shallow Specialization)
```yaml
Expert Configuration:
  type: shared_recurrent_engine
  shared_block: huginn_style
  num_recurrent_steps:
    train: 2-4 (randomized)
    inference: up to 16
  projections_per_expert:
    input: d_model→d_model
    output: d_model→d_model
  regularization:
    orthogonalize_projections: true
    projection_dropout: 0.0-0.1

Estimates:
  total_params: moderate (|R| + 2N×d_model^2)
  active_params_per_token: moderate
  throughput: moderate; good parameter efficiency
```

#### Synthesis II — MoE Layer as Recurrent Unit
```yaml
Layer Configuration:
  type: recurrent_moe_layer
  iterations:
    train: 2-4 (truncated BPTT)
    inference: 4-8
  residual_scale: 1.0
  routing_stabilization:
    min_entropy: 0.8
    logit_ema: 0.9
    z_loss_weight: 0.001

Estimates:
  total_params: similar to baseline MoE
  active_params_per_token: scales with iterations
  throughput: moderate; stable with proper damping
```

### 3.3 Training Pipeline

#### Phase 1: Pretraining (48 hours)
```python
# Pseudocode for training loop
def pretrain_more_model(model, config):
    # Dataset: WikiText-103 (~100M tokens)
    train_loader = create_wikitext_dataloader(
        batch_size=8,
        seq_length=512,
        shuffle=True
    )
    
    # Optimizer with DeepSpeed
    optimizer = DeepSpeedAdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate schedule
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=1e-5
    )
    
    # Training loop with monitoring
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Forward pass
            outputs = model(batch['input_ids'])
            loss = outputs.loss
            
            # Add auxiliary losses
            if hasattr(outputs, 'aux_loss'):
                loss += config.aux_loss_weight * outputs.aux_loss
            if hasattr(outputs, 'router_z_loss'):
                loss += config.router_z_loss_weight * outputs.router_z_loss
                
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            wandb.log({
                'loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'expert_usage': compute_expert_usage(outputs),
                'perplexity': torch.exp(loss).item()
            })
```

#### Phase 2: Fine-tuning (12 hours)
```python
def finetune_on_reasoning(model, config):
    # Multi-task on bAbI + mathematical reasoning
    datasets = {
        'babi': load_babi_tasks([1, 2, 3, 16]),  # Selected tasks
        'arithmetic': load_arithmetic_dataset(),
        'algebra': load_simple_algebra_dataset()
    }
    
    # Task-weighted sampling
    task_weights = {'babi': 0.4, 'arithmetic': 0.3, 'algebra': 0.3}
    
    # Lower learning rate for fine-tuning
    optimizer = DeepSpeedAdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    for step in range(num_finetune_steps):
        # Sample task
        task = sample_task(task_weights)
        batch = get_batch(datasets[task])
        
        # Task-specific processing
        if task == 'babi':
            # Format: context \n question \n answer
            outputs = model(batch['input_ids'])
            loss = ce_loss(outputs.logits, batch['labels'])
        else:
            # Chain-of-thought style for math
            outputs = model.generate_with_reasoning(
                batch['input_ids'],
                max_steps=config.max_reasoning_steps
            )
            loss = compute_reasoning_loss(outputs, batch['answers'])
            
        # Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### Phase 3: Evaluation (12 hours)
```python
def evaluate_model(model, config):
    results = {}
    
    # Language modeling evaluation
    wikitext_test = load_dataset('wikitext', 'wikitext-103-v1', split='test')
    results['perplexity'] = compute_perplexity(model, wikitext_test)
    
    # Reasoning evaluation
    for task_id in range(1, 21):  # All bAbI tasks
        task_data = load_babi_task(task_id, split='test')
        results[f'babi_task_{task_id}'] = evaluate_accuracy(model, task_data)
    
    # Arithmetic evaluation
    arithmetic_test = generate_arithmetic_test_set()
    results['arithmetic_accuracy'] = evaluate_arithmetic(model, arithmetic_test)
    
    # Commonsense reasoning
    hellaswag = load_dataset('hellaswag', split='validation')
    results['hellaswag_accuracy'] = evaluate_multiple_choice(model, hellaswag)
    
    # Expert analysis
    results['expert_specialization'] = analyze_expert_routing(model, validation_data)
    
    return results
```

### 3.4 Timeline

**Total Duration: 3 days (72 hours) per model**

```
Day 1 (0-24 hours): Initial Pretraining
├── Hours 0-6: Setup and debugging
│   ├── Environment setup with DeepSpeed
│   ├── Data preprocessing
│   └── Initial training runs to verify stability
├── Hours 6-18: Core pretraining
│   ├── WikiText-103 training
│   ├── Monitor convergence and expert usage
│   └── Adjust hyperparameters if needed
└── Hours 18-24: Checkpoint and analysis
    ├── Save model checkpoints
    ├── Analyze expert specialization
    └── Preliminary evaluation

Day 2 (24-48 hours): Continued Training
├── Hours 24-36: Advanced pretraining
│   ├── Continue WikiText training
│   ├── Gradually increase sequence length (curriculum)
│   └── Monitor for overfitting
├── Hours 36-42: Reasoning curriculum
│   ├── Introduce simple reasoning tasks
│   ├── Adjust recurrent depth (Option A)
│   └── Balance language modeling and reasoning
└── Hours 42-48: Fine-tuning preparation
    ├── Create task mixtures
    ├── Implement task-specific metrics
    └── Checkpoint before fine-tuning

Day 3 (48-72 hours): Fine-tuning and Evaluation
├── Hours 48-60: Multi-task fine-tuning
│   ├── bAbI tasks (focus on multi-hop)
│   ├── Arithmetic and simple algebra
│   └── Maintain language modeling quality
├── Hours 60-66: Comprehensive evaluation
│   ├── All test sets
│   ├── Expert behavior analysis
│   └── Ablation studies
└── Hours 66-72: Analysis and reporting
    ├── Generate visualizations
    ├── Statistical comparisons
    └── Prepare results summary
```

---

## 4. Datasets and Evaluation Methodology

### 4.1 Pretraining Dataset

**WikiText-103** [Merity et al., 2017]
- Size: 103M tokens
- Vocabulary: 267,735 unique tokens (we'll use BPE-32k)
- Average article length: 3.6k tokens
- Why chosen: Long-form text tests model's ability to maintain coherence

```python
# Data preprocessing
def prepare_wikitext_for_more():
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Or custom
    
    def tokenize_and_chunk(examples):
        # Tokenize
        tokenized = tokenizer(
            examples['text'],
            truncation=False,
            padding=False
        )
        
        # Chunk into sequences
        input_ids = []
        for ids in tokenized['input_ids']:
            # Add document boundaries
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            
            # Chunk with overlap for context
            for i in range(0, len(ids) - 512, 256):  # 50% overlap
                chunk = ids[i:i+512]
                if len(chunk) == 512:
                    input_ids.append(chunk)
                    
        return {'input_ids': input_ids}
    
    tokenized_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset
```

### 4.2 Reasoning Datasets

#### bAbI Tasks [Weston et al., 2015]
20 tasks testing different reasoning abilities:

```python
# Task descriptions and why they matter for MoRE
babi_tasks = {
    1: "Single Supporting Fact",     # Baseline memory
    2: "Two Supporting Facts",       # Multi-hop reasoning
    3: "Three Supporting Facts",     # Complex reasoning chains
    4: "Two Argument Relations",     # Relational reasoning
    5: "Three Argument Relations",   # Complex relations
    6: "Yes/No Questions",          # Binary classification
    7: "Counting",                  # Numerical reasoning
    8: "Lists/Sets",                # Set operations
    9: "Simple Negation",           # Logical negation
    10: "Indefinite Knowledge",     # Uncertainty handling
    11: "Basic Coreference",        # Pronoun resolution
    12: "Conjunction",              # Logical AND
    13: "Compound Coreference",     # Complex coreference
    14: "Time Reasoning",           # Temporal logic
    15: "Basic Deduction",          # Logical inference
    16: "Basic Induction",          # Pattern recognition
    17: "Positional Reasoning",     # Spatial reasoning
    18: "Size Reasoning",           # Comparative reasoning
    19: "Path Finding",             # Graph traversal
    20: "Agent Motivation"          # Goal reasoning
}

# Focus on tasks where recurrence should help
key_tasks_for_more = [2, 3, 7, 14, 15, 16, 19]  # Multi-step and pattern tasks
```

#### Arithmetic Dataset (Custom)
```python
def generate_arithmetic_problems():
    problems = []
    
    # Addition/Subtraction (2-4 digit numbers)
    for _ in range(1000):
        a = random.randint(10, 9999)
        b = random.randint(10, 9999)
        op = random.choice(['+', '-'])
        
        if op == '+':
            answer = a + b
        else:
            answer = a - b
            
        problems.append({
            'question': f"What is {a} {op} {b}?",
            'answer': str(answer),
            'type': 'arithmetic',
            'difficulty': len(str(max(a, b)))
        })
    
    # Simple word problems
    templates = [
        "If you have {a} apples and buy {b} more, how many do you have?",
        "A store had {a} items. They sold {b}. How many are left?",
        "You walked {a} steps yesterday and {b} steps today. Total?"
    ]
    
    for _ in range(500):
        template = random.choice(templates)
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        
        # Parse template to determine operation
        if "more" in template or "Total" in template:
            answer = a + b
        else:
            answer = abs(a - b)
            
        problems.append({
            'question': template.format(a=a, b=b),
            'answer': str(answer),
            'type': 'word_problem',
            'difficulty': 'medium'
        })
        
    return problems
```

#### Evaluation Metrics

```python
def compute_metrics(model, dataset, task_type):
    metrics = {
        'accuracy': 0,
        'perplexity': 0,
        'exact_match': 0,
        'f1_score': 0,
        'expert_entropy': 0,
        'reasoning_steps': []
    }
    
    for batch in dataset:
        outputs = model(batch['input_ids'])
        
        # Standard metrics
        predictions = outputs.logits.argmax(dim=-1)
        metrics['accuracy'] += (predictions == batch['labels']).float().mean()
        
        # Perplexity
        loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            batch['labels'].view(-1)
        )
        metrics['perplexity'] += torch.exp(loss)
        
        # MoE specific metrics
        if hasattr(outputs, 'router_logits'):
            # Expert usage entropy
            router_probs = F.softmax(outputs.router_logits, dim=-1)
            entropy = -(router_probs * router_probs.log()).sum(dim=-1).mean()
            metrics['expert_entropy'] += entropy
            
        # Reasoning depth (for Option A)
        if hasattr(outputs, 'num_reasoning_steps'):
            metrics['reasoning_steps'].append(outputs.num_reasoning_steps)
            
    # Average metrics
    num_batches = len(dataset)
    for key in metrics:
        if key != 'reasoning_steps':
            metrics[key] /= num_batches
            
    return metrics
```

---

## 5. Architecture Validation Metrics

### 5.1 Training Health Indicators

#### Loss and Convergence
```python
def monitor_training_health(metrics_history):
    indicators = {}
    
    # Smooth loss curve - should decrease monotonically
    loss_smooth = smooth(metrics_history['loss'], window=100)
    indicators['loss_variance'] = np.std(loss_smooth[-1000:])
    indicators['loss_trend'] = linear_regression_slope(loss_smooth[-1000:])
    
    # Gradient norms - should be stable
    grad_norms = metrics_history['grad_norm']
    indicators['grad_norm_mean'] = np.mean(grad_norms[-1000:])
    indicators['grad_norm_spikes'] = count_spikes(grad_norms, threshold=10.0)
    
    # Learning rate effectiveness
    lr_changes = metrics_history['learning_rate']
    loss_changes = np.diff(metrics_history['loss'])
    indicators['lr_correlation'] = np.corrcoef(lr_changes[:-1], loss_changes)[0, 1]
    
    return indicators

# Warning thresholds
health_thresholds = {
    'loss_variance': 0.1,      # High variance indicates instability
    'grad_norm_spikes': 10,    # More than 10 spikes per 1000 steps
    'lr_correlation': -0.1     # LR changes should correlate with loss reduction
}
```

#### Expert Utilization Metrics
```python
def analyze_expert_usage(router_outputs, num_experts=8):
    # router_outputs: [batch_size, seq_len, num_experts]
    
    # Expert load distribution
    expert_counts = torch.zeros(num_experts)
    router_probs = F.softmax(router_outputs, dim=-1)
    
    # Count tokens per expert (soft assignment)
    expert_counts = router_probs.sum(dim=[0, 1])
    total_tokens = router_probs.size(0) * router_probs.size(1)
    
    # Compute balance metrics
    expert_load = expert_counts / total_tokens
    
    metrics = {
        'expert_load': expert_load.tolist(),
        'load_balance_cv': expert_load.std() / expert_load.mean(),  # Coefficient of variation
        'max_load_ratio': expert_load.max() / expert_load.mean(),
        'min_load_ratio': expert_load.min() / expert_load.mean(),
        'effective_experts': 1 / (expert_load ** 2).sum(),  # Inverse HHI
        'router_entropy': -(router_probs * router_probs.log()).sum(dim=-1).mean()
    }
    
    # Specialization analysis
    if hasattr(router_outputs, 'token_types'):
        # Analyze which experts handle which token types
        specialization = compute_expert_specialization(
            router_probs, 
            router_outputs.token_types
        )
        metrics['specialization_score'] = specialization
        
    return metrics

# Healthy ranges
expert_health = {
    'load_balance_cv': (0.0, 0.3),      # Lower is better
    'max_load_ratio': (0.8, 1.5),       # Near 1.0 is ideal
    'min_load_ratio': (0.5, 1.2),       # Not too low
    'effective_experts': (6.0, 8.0),     # For 8 experts total
    'router_entropy': (1.5, 2.8)        # log(8) ≈ 2.08 is uniform
}
```

### 5.2 Architecture-Specific Indicators

#### Synthesis I (Recurrent Experts): Recurrent Depth Utilization
```python
def analyze_recurrent_depth(model_outputs):
    """Specific to Option A - analyze if recurrence is being used effectively"""
    
    metrics = {}
    
    # Distribution of recurrent steps used
    if hasattr(model_outputs, 'recurrent_steps_used'):
        steps = model_outputs.recurrent_steps_used
        metrics['mean_steps'] = steps.float().mean()
        metrics['std_steps'] = steps.float().std()
        metrics['max_steps'] = steps.max()
        
    # Change in hidden states across steps
    if hasattr(model_outputs, 'hidden_states_per_step'):
        states = model_outputs.hidden_states_per_step  # [steps, batch, seq, hidden]
        
        # Measure how much states change between steps
        state_changes = []
        for i in range(len(states) - 1):
            change = (states[i+1] - states[i]).norm(dim=-1).mean()
            state_changes.append(change.item())
            
        metrics['mean_state_change'] = np.mean(state_changes)
        metrics['state_change_trend'] = linear_regression_slope(state_changes)
        
        # Check if later steps contribute
        final_contribution = (states[-1] - states[-2]).norm() / states[-1].norm()
        metrics['final_step_contribution'] = final_contribution.item()
        
    return metrics

# Healthy indicators for Synthesis I (Recurrent Experts)
option_synthI_recurrent_health = {
    'mean_steps': (2.0, 8.0),           # Using multiple steps
    'mean_state_change': (0.1, 2.0),    # States are changing
    'state_change_trend': (-0.1, 0.1),  # Relatively stable
    'final_step_contribution': (0.05, 0.5)  # Last step matters
}
```

#### Synthesis I (Shared-Block): Projection Behavior
```python
def analyze_shared_block_expert(model_outputs):
    """Analyze specialization emerging from expert projections around shared engine."""
    
    metrics = {}
    
    if hasattr(model_outputs, 'projection_stats'):
        stats = model_outputs.projection_stats
        metrics['proj_orthogonality'] = stats['orthogonality_mean']
        metrics['proj_diversity'] = stats['pairwise_divergence_mean']
        metrics['shared_core_util'] = stats['core_activation_norm']
        
    return metrics
```

#### Synthesis II (MoE Layer Recurrence): Iterative Routing Stability
```python
def analyze_moe_recurrence(model_outputs):
    """Analyze routing behavior across MoE iterations."""
    
    metrics = {}
    
    if hasattr(model_outputs, 'routing_logits_per_iter'):
        logits_list = model_outputs.routing_logits_per_iter  # list over iterations
        entropies = []
        max_probs = []
        for logits in logits_list:
            probs = F.softmax(logits, dim=-1)
            entropies.append(-(probs * probs.log()).sum(dim=-1).mean())
            max_probs.append(probs.max(dim=-1)[0].mean())
        metrics['routing_entropy_mean'] = torch.stack(entropies).mean()
        metrics['routing_maxprob_trend'] = (torch.tensor(max_probs[-1]) - torch.tensor(max_probs[0]))
            
    return metrics

# Healthy ranges for Option C
option_synthII_moe_health = {
    'routing_entropy_mean': (1.2, 2.5),   # Avoid collapse/chaos
    'routing_maxprob_trend': (0.0, 0.3),  # Confidence grows moderately
}
```

#### Option D: Refinement Effectiveness
```python
def analyze_refinement(model_outputs):
    """Specific to Option D - analyze if refinement improves outputs"""
    
    metrics = {}
    
    if hasattr(model_outputs, 'before_refinement') and hasattr(model_outputs, 'after_refinement'):
        before = model_outputs.before_refinement
        after = model_outputs.after_refinement
        
        # Magnitude of refinement
        refinement_delta = (after - before).norm(dim=-1)
        metrics['refinement_magnitude'] = refinement_delta.mean()
        metrics['refinement_relative'] = (refinement_delta / before.norm(dim=-1)).mean()
        
        # Direction analysis - is refinement consistent?
        if hasattr(model_outputs, 'refinement_steps'):
            steps = model_outputs.refinement_steps
            
            # Cosine similarity between refinement directions
            cos_sim = F.cosine_similarity(
                steps[1] - steps[0],
                steps[2] - steps[1],
                dim=-1
            )
            metrics['refinement_consistency'] = cos_sim.mean()
            
    return metrics
```

### 5.3 Comparative Analysis Framework

```python
def compare_architectures(results_dict):
    """Compare all architectures on key metrics"""
    
    comparison = pd.DataFrame()
    
    for model_name, results in results_dict.items():
        row = {
            'model': model_name,
            'total_params': results['total_params'],
            'active_params': results['active_params_per_token'],
            
            # Performance metrics
            'wiki_perplexity': results['wiki_perplexity'],
            'babi_avg_accuracy': np.mean([results[f'babi_{i}'] for i in range(1, 21)]),
            'babi_multihop_accuracy': np.mean([results[f'babi_{i}'] for i in [2, 3, 16]]),
            'arithmetic_accuracy': results['arithmetic_accuracy'],
            'hellaswag_accuracy': results['hellaswag_accuracy'],
            
            # Efficiency metrics
            'tokens_per_second': results['throughput'],
            'memory_usage_gb': results['peak_memory_gb'],
            'training_time_hours': results['training_time'],
            
            # MoE health
            'expert_usage_cv': results['expert_metrics']['load_balance_cv'],
            'effective_experts': results['expert_metrics']['effective_experts'],
            
            # Architecture-specific
            'architecture_health_score': compute_health_score(results)
        }
        
        comparison = comparison.append(row, ignore_index=True)
        
    # Compute relative improvements
    baseline_idx = comparison[comparison['model'] == 'Standard_MoE'].index[0]
    for col in ['wiki_perplexity', 'babi_avg_accuracy', 'arithmetic_accuracy']:
        comparison[f'{col}_improvement'] = (
            (comparison[col] - comparison.loc[baseline_idx, col]) / 
            comparison.loc[baseline_idx, col] * 100
        )
        
    return comparison

def compute_health_score(results):
    """Aggregate health score for each architecture"""
    score = 100.0
    
    # Deduct points for unhealthy metrics
    if results['expert_metrics']['load_balance_cv'] > 0.3:
        score -= 10
    if results['gradient_norm_spikes'] > 10:
        score -= 15
    if results['training_divergences'] > 0:
        score -= 25
        
    # Add points for good behavior
    if results['expert_metrics']['effective_experts'] > 6:
        score += 5
    if results.get('refinement_effectiveness', 0) > 0.1:
        score += 10
        
    return max(0, min(100, score))
```

---

## 6. Technology Stack and Tools

### 6.1 Core Framework

**PyTorch 2.0+**
- Primary deep learning framework
- Key features used:
  - `torch.compile()` for optimized kernels
  - `torch.nn.attention.flex_attention` for custom attention patterns
  - Mixed precision training with `torch.cuda.amp`
  - Custom CUDA kernels for recurrent operations

```python
# Example of torch.compile usage for MoRE
import torch
from torch import nn

@torch.compile(mode="reduce-overhead")
def optimized_recurrent_expert(x, expert_weights, num_steps):
    """Compiled version of recurrent expert forward pass"""
    hidden = x
    
    for _ in range(num_steps):
        # Optimized recurrent computation
        hidden = torch.tanh(
            torch.matmul(hidden, expert_weights['W_hh']) + 
            torch.matmul(x, expert_weights['W_ih']) +
            expert_weights['bias']
        )
        
    return hidden
```

### 6.2 Distributed Training

**DeepSpeed**
- Efficient training of large models
- Key features:
  - ZeRO-2 optimizer state partitioning
  - CPU offloading for optimizer states
  - MoE layer implementation
  - Activation checkpointing

```python
# DeepSpeed configuration
deepspeed_config = {
    "train_batch_size": 48,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 6,
    
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e12,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4
    },
    
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": 4,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    },
    
    "moe": {
        "enabled": True,
        "ep_size": 1,
        "num_experts": 8,
        "top_k": 1,
        "min_capacity": 4,
        "noisy_gate_policy": "Jitter",
        "moe_param_group": True
    }
}
```

### 6.3 Monitoring and Visualization

**Weights & Biases (WandB)**
```python
import wandb

# Initialize wandb
wandb.init(
    project="more-experiments",
    config={
        "architecture": "synthesis_i_shared" ,  # or synthesis_i_recurrent, synthesis_ii_layer
        "num_experts": 8,
        "recurrent_steps": 4,
        "learning_rate": 2e-4,
        "batch_size": 48
    }
)

# Custom logging for MoRE
class MoRELogger:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def log_step(self, metrics, step):
        # Standard metrics
        wandb.log({
            f"{self.model_name}/loss": metrics['loss'],
            f"{self.model_name}/perplexity": metrics['perplexity'],
            f"{self.model_name}/learning_rate": metrics['lr'],
            f"{self.model_name}/grad_norm": metrics['grad_norm'],
        }, step=step)
        
        # Expert metrics
        expert_data = []
        for i, load in enumerate(metrics['expert_loads']):
            expert_data.append({
                "expert_id": i,
                "load_fraction": load,
                "tokens_routed": metrics['expert_tokens'][i]
            })
        
        wandb.log({
            f"{self.model_name}/expert_table": wandb.Table(
                dataframe=pd.DataFrame(expert_data)
            )
        }, step=step)
        
        # Architecture-specific metrics
        if 'recurrent_steps' in metrics:
            wandb.log({
                f"{self.model_name}/mean_recurrent_steps": metrics['recurrent_steps'].mean(),
                f"{self.model_name}/recurrent_histogram": wandb.Histogram(metrics['recurrent_steps'])
            }, step=step)
```

**TensorBoard** (Backup/Local)
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/more_{model_name}')

# Expert routing visualization
def visualize_expert_routing(writer, routing_probs, step):
    # routing_probs: [batch, seq_len, num_experts]
    
    # Create heatmap of expert selection
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average routing probabilities
    avg_routing = routing_probs.mean(dim=[0, 1]).cpu().numpy()
    
    # Plot
    ax.bar(range(len(avg_routing)), avg_routing)
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Average Routing Probability')
    ax.set_title(f'Expert Usage Distribution (Step {step})')
    
    writer.add_figure('expert_routing/distribution', fig, step)
    plt.close()
    
    # Routing entropy over time
    entropy = -(routing_probs * routing_probs.log()).sum(dim=-1).mean()
    writer.add_scalar('expert_routing/entropy', entropy, step)
```

### 6.4 Development Tools

**Environment Setup**
```bash
# Create conda environment
conda create -n more python=3.10
conda activate more

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install deepspeed>=0.12.0
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install wandb
pip install tensorboard
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tqdm
pip install sentencepiece  # For tokenization

# Development tools
pip install black isort flake8  # Code formatting
pip install pytest pytest-cov   # Testing
pip install jupyter notebook    # Experimentation
```

**VS Code Extensions**
- Python
- Pylance
- Jupyter
- Remote-SSH (for cloud training)
- GitLens
- Markdown All in One

### 6.5 Hardware Profiling

**NVIDIA Tools**
```python
# GPU memory profiling
import torch.cuda

def profile_model_memory(model, batch_size=8, seq_len=512):
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len)).cuda()
    
    # Measure initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Forward pass
    with torch.cuda.amp.autocast():
        outputs = model(dummy_input)
        loss = outputs.loss
        
    forward_memory = torch.cuda.memory_allocated() / 1024**3
    
    # Backward pass
    loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    return {
        'initial_memory_gb': initial_memory,
        'forward_memory_gb': forward_memory,
        'peak_memory_gb': peak_memory,
        'activation_memory_gb': forward_memory - initial_memory
    }

# Throughput profiling
def profile_throughput(model, batch_size=8, seq_len=512, num_steps=100):
    model.eval()
    
    # Warmup
    for _ in range(10):
        dummy_input = torch.randint(0, 32000, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Timing
    torch.cuda.synchronize()
    start = time.time()
    
    total_tokens = 0
    for _ in range(num_steps):
        dummy_input = torch.randint(0, 32000, (batch_size, seq_len)).cuda()
        with torch.no_grad():
            _ = model(dummy_input)
        total_tokens += batch_size * seq_len
        
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return {
        'tokens_per_second': total_tokens / elapsed,
        'seconds_per_batch': elapsed / num_steps,
        'gpu_utilization': get_gpu_utilization()  # Custom function using nvidia-ml-py
    }
```

---

## 7. Technical Implementation Details

### 7.1 Model Architecture Implementation

**Base MoRE Model Class**
```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class MoREConfig:
    # Model dimensions
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    
    # MoE configuration
    num_experts: int = 8
    expert_type: str = "recurrent"  # One of: standard, recurrent, sandwich, parallel, sequential
    top_k: int = 1
    capacity_factor: float = 1.0
    
    # Recurrent configuration
    recurrent_type: str = "huginn"  # One of: gru, lstm, huginn
    num_recurrent_steps: int = 4
    recurrent_hidden_size: Optional[int] = None
    
    # Training configuration
    dropout: float = 0.1
    aux_loss_weight: float = 0.001
    router_z_loss_weight: float = 0.001
    
    def __post_init__(self):
        if self.recurrent_hidden_size is None:
            self.recurrent_hidden_size = self.d_model

class MoREModel(nn.Module):
    def __init__(self, config: MoREConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Positional encoding
        self.pos_encoder = RotaryPositionalEncoding(config.d_model, config.n_heads)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MoRELayer(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(input_ids.shape[1], hidden_states.device)
            
        # Apply transformer layers
        all_router_logits = []
        all_expert_usage = []
        
        for layer in self.layers:
            hidden_states, router_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            
            if router_outputs is not None:
                all_router_logits.append(router_outputs['logits'])
                all_expert_usage.append(router_outputs['expert_usage'])
                
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute losses
        loss = None
        if input_ids is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary losses
            if all_router_logits:
                # Load balancing loss
                aux_loss = compute_load_balancing_loss(all_expert_usage)
                loss = loss + self.config.aux_loss_weight * aux_loss
                
                # Router z-loss
                z_loss = compute_router_z_loss(all_router_logits)
                loss = loss + self.config.router_z_loss_weight * z_loss
                
        if return_dict:
            return MoREOutput(
                loss=loss,
                logits=logits,
                router_logits=all_router_logits,
                expert_usage=all_expert_usage,
            )
        else:
            return (loss, logits)
```

### 7.2 MoE Layer with Recurrent Experts

```python
class MoRELayer(nn.Module):
    def __init__(self, config: MoREConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.attn_ln = RMSNorm(config.d_model)
        
        # MoE layer
        self.moe = MoELayer(config)
        self.moe_ln = RMSNorm(config.d_model)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_ln(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE layer with residual
        residual = hidden_states
        hidden_states = self.moe_ln(hidden_states)
        hidden_states, router_outputs = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, router_outputs

class MoELayer(nn.Module):
    def __init__(self, config: MoREConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = Router(
            config.d_model,
            config.num_experts,
            config.top_k,
            noise_policy='jitter' if self.training else None
        )
        
        # Create experts based on type
        if config.expert_type == "standard":
            expert_class = StandardFFNExpert
        elif config.expert_type == "recurrent":
            expert_class = RecurrentExpert
        elif config.expert_type == "sandwich":
            expert_class = SandwichRecurrentExpert
        elif config.expert_type == "parallel":
            expert_class = ParallelBranchExpert
        elif config.expert_type == "sequential":
            expert_class = SequentialRecurrentExpert
        else:
            raise ValueError(f"Unknown expert type: {config.expert_type}")
            
        self.experts = nn.ModuleList([
            expert_class(config) for _ in range(config.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Flatten batch and sequence dimensions for routing
        hidden_states_flat = hidden_states.view(-1, d_model)
        
        # Route tokens to experts
        router_output = self.router(hidden_states_flat)
        
        # Process tokens through selected experts
        expert_outputs = torch.zeros_like(hidden_states_flat)
        
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = router_output['expert_masks'][expert_idx]
            
            if expert_mask.any():
                expert_input = hidden_states_flat[expert_mask]
                expert_out = expert(expert_input)
                
                # Apply routing weights
                weights = router_output['routing_weights'][expert_mask, expert_idx:expert_idx+1]
                expert_outputs[expert_mask] = expert_out * weights
                
        # Reshape back to original dimensions
        output = expert_outputs.view(batch_size, seq_len, d_model)
        
        return output, router_output
```

### 7.3 Implementing the Huginn-Style Recurrent Block

```python
class HuginnRecurrentBlock(nn.Module):
    """
    Implementation of the recurrent reasoning block from Huginn model.
    Based on https://ollama.hf-mirror.com/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Input adapter - merges static input with evolving state
        self.adapter = nn.Linear(2 * self.hidden_size, self.hidden_size)
        
        # Core transformer-like block
        self.ln1 = RMSNorm(self.hidden_size)
        self.self_attn = FlexAttention(
            self.hidden_size,
            self.num_heads,
            self.head_dim,
            use_bias=False
        )
        self.ln2 = RMSNorm(self.hidden_size)
        
        # Gated MLP (SwiGLU style)
        self.ln3 = RMSNorm(self.hidden_size)
        self.mlp = GatedMLP(
            self.hidden_size,
            intermediate_size=4 * self.hidden_size,
            activation='silu'
        )
        self.ln4 = RMSNorm(self.hidden_size)
        
        # Recurrence control
        self.num_steps = config.num_recurrent_steps
        
    def forward(self, x, num_steps=None):
        """
        Args:
            x: Input tensor [batch_size, d_model]
            num_steps: Number of recurrent steps (overrides config if provided)
        
        Returns:
            output: Final state after recurrence [batch_size, d_model]
            intermediate_states: List of states at each step (optional)
        """
        if num_steps is None:
            num_steps = self.num_steps
            
        batch_size = x.size(0)
        device = x.device
        
        # Initialize latent state (random as in Huginn)
        state = torch.randn(batch_size, self.hidden_size, device=device) * 0.02
        static_input = x.clone()  # Keep original input
        
        intermediate_states = []
        
        for step in range(num_steps):
            # Adapt current state with static input
            combined = torch.cat([state, static_input], dim=-1)
            state = self.adapter(combined)
            
            # Self-attention
            residual = state
            state = self.ln1(state)
            
            # For single-token expert, self-attention might be simplified
            # In full implementation, this would attend over sequence
            attn_out, _ = self.self_attn(
                state.unsqueeze(1),  # Add sequence dimension
                state.unsqueeze(1),
                state.unsqueeze(1),
            )
            state = attn_out.squeeze(1)  # Remove sequence dimension
            
            state = self.ln2(residual + state)
            
            # Gated MLP
            residual = state
            state = self.ln3(state)
            state = self.mlp(state)
            state = self.ln4(residual + state)
            
            intermediate_states.append(state)
            
        return state, intermediate_states

class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP as used in Huginn"""
    
    def __init__(self, hidden_size, intermediate_size, activation='silu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Single projection to 2x intermediate size
        self.gate_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation
        self.act = nn.SiLU() if activation == 'silu' else nn.GELU()
        
    def forward(self, x):
        # Project and split
        gate_output = self.gate_proj(x)
        x1, x2 = gate_output.chunk(2, dim=-1)
        
        # Gated activation
        hidden = self.act(x1) * x2
        
        # Project back
        return self.down_proj(hidden)
```

### 7.4 Training Loop with Curriculum

```python
class MoRETrainer:
    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
            model_parameters=model.parameters()
        )
        
        # Curriculum settings
        self.curriculum_stage = 0
        self.curriculum_schedule = {
            0: {'seq_len': 128, 'recurrent_steps': 1},
            10000: {'seq_len': 256, 'recurrent_steps': 2},
            20000: {'seq_len': 512, 'recurrent_steps': 4},
            30000: {'seq_len': 512, 'recurrent_steps': 8},
        }
        
    def train_step(self, batch, step):
        # Update curriculum
        self.update_curriculum(step)
        
        # Adjust batch for current curriculum
        batch = self.adjust_batch_for_curriculum(batch)
        
        # Forward pass
        outputs = self.model_engine(batch['input_ids'])
        
        # Compute loss with gradient accumulation
        loss = outputs.loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.model_engine.backward(loss)
        
        # Optimizer step
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            self.model_engine.step()
            
        # Logging
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'perplexity': torch.exp(loss).item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        # Add MoE metrics
        if hasattr(outputs, 'expert_usage'):
            metrics.update(self.compute_expert_metrics(outputs.expert_usage))
            
        return metrics
        
    def update_curriculum(self, step):
        """Update training configuration based on curriculum"""
        for threshold, settings in sorted(self.curriculum_schedule.items()):
            if step >= threshold:
                if hasattr(self.model.module, 'config'):
                    # Update model configuration
                    for key, value in settings.items():
                        if hasattr(self.model.module.config, key):
                            setattr(self.model.module.config, key, value)
                            
    def compute_expert_metrics(self, expert_usage):
        """Compute expert load balancing metrics"""
        # Aggregate usage across layers
        total_usage = torch.zeros(self.config.num_experts)
        
        for layer_usage in expert_usage:
            total_usage += layer_usage.sum(dim=0)
            
        # Normalize
        total_usage = total_usage / total_usage.sum()
        
        # Compute metrics
        metrics = {
            'expert_load_cv': total_usage.std() / total_usage.mean(),
            'expert_load_max': total_usage.max().item(),
            'expert_load_min': total_usage.min().item(),
            'effective_experts': 1.0 / (total_usage ** 2).sum().item(),
        }
        
        # Per-expert usage
        for i, usage in enumerate(total_usage):
            metrics[f'expert_{i}_load'] = usage.item()
            
        return metrics
```

### 7.5 Evaluation Pipeline

```python
class MoREEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    @torch.no_grad()
    def evaluate_perplexity(self, dataset, max_samples=1000):
        """Evaluate language modeling perplexity"""
        total_loss = 0
        total_tokens = 0
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
                
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).cuda()
            
            outputs = self.model(input_ids)
            
            # Compute loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            
            total_loss += loss.sum().item()
            total_tokens += loss.numel()
            
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
        
    @torch.no_grad()
    def evaluate_reasoning(self, dataset, task_type='babi'):
        """Evaluate on reasoning tasks"""
        correct = 0
        total = 0
        
        for sample in dataset:
            if task_type == 'babi':
                # Format: context \n question \n answer
                prompt = f"{sample['context']}\n{sample['question']}\n"
                target = sample['answer']
                
            elif task_type == 'arithmetic':
                prompt = f"Question: {sample['question']}\nAnswer:"
                target = sample['answer']
                
            # Generate answer
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
            
            # For recurrent models, we might want to use more steps
            if hasattr(self.model.module, 'config') and self.model.module.config.expert_type == 'recurrent':
                # Temporarily increase recurrent steps for reasoning
                original_steps = self.model.module.config.num_recurrent_steps
                self.model.module.config.num_recurrent_steps = 8
                
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Reset recurrent steps
            if hasattr(self.model.module, 'config') and self.model.module.config.expert_type == 'recurrent':
                self.model.module.config.num_recurrent_steps = original_steps
                
            # Decode and check answer
            generated = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
            
            # Extract answer (before newline or period)
            answer = generated.split('\n')[0].split('.')[0].strip()
            
            if answer.lower() == target.lower():
                correct += 1
            total += 1
            
        accuracy = correct / total if total > 0 else 0
        return accuracy
        
    def evaluate_expert_specialization(self, dataset, num_samples=1000):
        """Analyze what types of inputs each expert handles"""
        expert_token_types = defaultdict(lambda: defaultdict(int))
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).cuda()
            
            # Get router outputs
            with torch.no_grad():
                _, router_outputs = self.model(input_ids, return_router_outputs=True)
                
            # Analyze token types routed to each expert
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            
            for layer_idx, layer_routing in enumerate(router_outputs):
                expert_assignments = layer_routing['expert_masks']
                
                for token_idx, token in enumerate(tokens):
                    # Categorize token
                    if token.isdigit():
                        token_type = 'numeric'
                    elif token in ['.', ',', '!', '?']:
                        token_type = 'punctuation'
                    elif token.startswith('##'):
                        token_type = 'subword'
                    elif len(token) == 1:
                        token_type = 'single_char'
                    else:
                        token_type = 'word'
                        
                    # Record which expert handled this token type
                    for expert_idx in range(self.model.module.config.num_experts):
                        if expert_assignments[expert_idx][token_idx]:
                            expert_token_types[expert_idx][token_type] += 1
                            
        return dict(expert_token_types)
```

### 7.6 Memory Optimization Techniques

```python
def optimize_memory_usage(model, config):
    """Apply memory optimization techniques for single GPU training"""
    
    # 1. Gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        # Manual implementation for custom layers
        for layer in model.layers:
            layer.use_checkpoint = True
            
    # 2. Parameter sharing where applicable
    if config.share_embeddings:
        # Tie input and output embeddings
        model.lm_head.weight = model.embed_tokens.weight
        
    # 3. Mixed precision setup
    model = model.half()  # Convert to fp16
    
    # Keep critical components in fp32
    for module in model.modules():
        if isinstance(module, (Router, RMSNorm)):
            module.float()
            
    # 4. Efficient attention implementation
    if config.use_flash_attention:
        replace_attention_with_flash_attention(model)
        
    # 5. CPU offloading for inactive experts
    if config.expert_offloading:
        setup_expert_offloading(model, config)
        
    return model

def setup_expert_offloading(model, config):
    """Setup CPU offloading for experts not in use"""
    
    class OffloadedExpert(nn.Module):
        def __init__(self, expert):
            super().__init__()
            self.expert = expert
            self.on_gpu = False
            
        def to_gpu(self):
            if not self.on_gpu:
                self.expert = self.expert.cuda()
                self.on_gpu = True
                
        def to_cpu(self):
            if self.on_gpu:
                self.expert = self.expert.cpu()
                self.on_gpu = False
                
        def forward(self, x):
            self.to_gpu()
            output = self.expert(x)
            # Offload after use if memory pressure
            if torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                self.to_cpu()
            return output
            
    # Wrap experts
    for layer in model.layers:
        if hasattr(layer, 'moe'):
            layer.moe.experts = nn.ModuleList([
                OffloadedExpert(expert) for expert in layer.moe.experts
            ])
```

### 7.7 Advanced Monitoring and Debugging

```python
class MoREDebugger:
    """Advanced debugging tools for MoRE training"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.activation_hooks = []
        self.gradient_hooks = []
        
    def register_hooks(self):
        """Register forward and backward hooks for monitoring"""
        
        def forward_hook(module, input, output, name):
            # Store activation statistics
            if isinstance(output, torch.Tensor):
                self.log_tensor_stats(f"{name}_output", output)
            
        def backward_hook(module, grad_input, grad_output, name):
            # Store gradient statistics
            if grad_output[0] is not None:
                self.log_tensor_stats(f"{name}_grad", grad_output[0])
                
        # Register hooks on key modules
        for name, module in self.model.named_modules():
            if isinstance(module, (RecurrentExpert, Router, GatedMLP)):
                handle_fwd = module.register_forward_hook(
                    lambda m, i, o, n=name: forward_hook(m, i, o, n)
                )
                handle_bwd = module.register_backward_hook(
                    lambda m, gi, go, n=name: backward_hook(m, gi, go, n)
                )
                
                self.activation_hooks.append(handle_fwd)
                self.gradient_hooks.append(handle_bwd)
                
    def log_tensor_stats(self, name, tensor):
        """Log statistics about a tensor"""
        stats = {
            f"{name}_mean": tensor.mean().item(),
            f"{name}_std": tensor.std().item(),
            f"{name}_max": tensor.max().item(),
            f"{name}_min": tensor.min().item(),
            f"{name}_norm": tensor.norm().item(),
        }
        
        # Check for anomalies
        if torch.isnan(tensor).any():
            stats[f"{name}_has_nan"] = True
        if torch.isinf(tensor).any():
            stats[f"{name}_has_inf"] = True
            
        wandb.log(stats)
        
    def analyze_expert_gradients(self):
        """Analyze gradient flow through experts"""
        expert_grads = {}
        
        for name, param in self.model.named_parameters():
            if 'expert' in name and param.grad is not None:
                expert_idx = int(name.split('experts.')[1].split('.')[0])
                
                if expert_idx not in expert_grads:
                    expert_grads[expert_idx] = []
                    
                expert_grads[expert_idx].append(param.grad.norm().item())
                
        # Compute statistics per expert
        for expert_idx, grads in expert_grads.items():
            wandb.log({
                f"expert_{expert_idx}_grad_norm": np.mean(grads),
                f"expert_{expert_idx}_grad_std": np.std(grads),
            })
            
    def check_router_behavior(self, num_batches=10):
        """Analyze router behavior over multiple batches"""
        router_stats = defaultdict(list)
        
        for _ in range(num_batches):
            # Get a batch
            batch = next(iter(self.train_loader))
            
            with torch.no_grad():
                _, router_outputs = self.model(
                    batch['input_ids'].cuda(),
                    return_router_outputs=True
                )
                
            # Analyze routing patterns
            for layer_idx, routing in enumerate(router_outputs):
                # Compute entropy
                probs = routing['routing_weights']
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                router_stats[f'layer_{layer_idx}_entropy'].append(entropy.item())
                
                # Check for collapsed routing
                max_prob = probs.max(dim=-1)[0].mean()
                router_stats[f'layer_{layer_idx}_max_prob'].append(max_prob.item())
                
        # Report findings
        for metric, values in router_stats.items():
            print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
```

---

## Summary

This comprehensive guide covers the complete MoRE (Mixture of Recurrent Experts) research project, from conceptual motivation through detailed implementation. The key contributions are:

1. **Novel Architecture**: Combining MoE's parameter efficiency with recurrent processing's computational depth
2. **Synthesis Variants**: Two families — Synthesis I (Recurrent Experts and Shared-Block Experts) and Synthesis II (MoE-as-Recurrent-Layer)
3. **Practical Implementation**: Designed for single-GPU training with careful memory management
4. **Comprehensive Evaluation**: Covering both language modeling and reasoning capabilities
5. **Detailed Monitoring**: Extensive metrics for validating architectural effectiveness

The research aims to demonstrate that recurrent experts can outperform standard FFN experts in reasoning tasks while maintaining competitive language modeling performance, potentially opening new directions for efficient large language model design.

# MoRE: Mixture of Recurrent Experts - Research and Validation Plan

## 1. Introduction: From MoE to MoRE

### 1.1 The Rise of Sparsity: Mixture-of-Experts (MoE)
Modern large language models (LLMs) have demonstrated incredible capabilities, largely driven by scaling up model size and training data. However, training and serving these dense, monolithic models incurs substantial computational costs. A promising approach to mitigate this is through **Sparsely-Gated Mixture-of-Experts (MoE)** models.

The MoE architecture, first introduced to NLP at scale by Shazeer et al. (2017), replaces dense feed-forward network (FFN) layers with a set of parallel "expert" networks. For each input token, a lightweight, trainable "router" network selects one or a few experts to process it. This conditional computation means that while the total number of parameters in the model can be enormous, the computational cost (FLOPs) per token remains constant. This allows for scaling model capacity without a proportional increase in inference or training cost.

Recent work, such as the **Switch Transformer** (Fedus et al., 2021), simplified this paradigm by routing each token to only a single expert (top-1 routing), further improving training efficiency and demonstrating that sparse models can significantly outperform dense models with the same computational budget.

### 1.2 The Next Step: Mixture of Recurrent Experts (MoRE)
While standard MoE models scale parameters efficiently by making FFNs sparse, the experts themselves are typically simple, static, two-layer feed-forward networks. They perform a one-shot transformation on the input token.

I propose the **Mixture of Recurrent Experts (MoRE)**, a novel architecture that enhances MoE by replacing or augmenting the FFN experts with **recurrent blocks**. The core hypothesis is that equipping experts with recurrence will enable them to perform more complex, iterative computations on a per-token basis. This could unlock or improve capabilities that are challenging for standard Transformers, such as:

*   **Multi-step Reasoning:** Solving problems that require sequential steps of logic or calculation.
*   **Algorithmic Tasks:** Executing simple algorithms internally.
*   **Adaptive Computation:** Spending more computational effort on "harder" tokens by unrolling the recurrence for more steps.

This concept draws inspiration from models that leverage weight-tying and recurrence to simulate greater depth and computational power, such as the Universal Transformer and recent work on latent reasoning models like **Huginn** (Geiping et al., 2025), which achieve state-of-the-art reasoning performance by iterating a single recurrent block. By integrating this "iterative refinement" capability into the MoE framework, MoRE aims to create models that are both parameter-efficient and computationally powerful.

This document outlines a plan to explore, implement, and validate three MoRE variants across two synthesis families against a standard MoE baseline.

---

## 2. MoRE Architecture Variants (Synthesis I & II)

We will investigate three primary designs for integrating recurrence, categorized by the level at which recurrence is applied: the individual expert level (Synthesis I) or the entire MoE layer level (Synthesis II).

### **Critical Architecture Distinction: Depth vs. Layers**

**Important**: The MoRE variants represent a fundamental shift in how depth is achieved:

- **Traditional Approach**: Stack multiple transformer layers (6 layers in our baseline)
- **MoRE Approach**: Use **1 transformer layer** where the **recurrent block gets repeated multiple times** to create depth

This means:
- **Base MoE**: 6 separate computational layers, each with MoE
- **MoRE Variants**: 1 computational layer, but the recurrent block inside gets reused r times, creating r× computational depth within that single layer

The key insight is that **recurrence creates depth through iteration rather than through layer stacking**.

### Synthesis I: Recurrence at the Expert Level

This family of architectures integrates recurrence at the most granular level: within the experts of an MoE layer. This approach conceptualizes the experts not as simple non-linear transformations but as self-contained reasoning modules.

#### Option A: Independent Recurrent Experts
* **Architecture & Layer Order:** In this design, each of the N experts in an MoE layer is a complete, independent, Huginn-style recurrent block. When the router selects an expert, the token is processed by that expert for a fixed number of internal recurrence steps (`r`). This embodies a **"deep specialization"** model, where each expert can learn a unique, complex internal algorithm.
* **Token Flow & Data Shapes:** A token `x` (shape `[d_model]`) is routed to an expert. The expert initializes a latent state `s_0` (e.g., from `x`) and iteratively computes `s_i = R_expert(x, s_{i-1})` for `r` steps. The final state `s_r` is the expert's output.
* **Parameter & FLOP Profile:** This is the most computationally expensive design. The parameter count is high, as each of the N experts has its own full set of weights for its recurrent block. The FLOPs are multiplied by the number of experts selected (`k`) and the recurrence steps (`r`), making it most feasible for top-1 routing (`k=1`) and a small `r`.
* **Strengths:**
    * **Maximum Specialization:** Allows for highly sophisticated, domain-specific "algorithms" to develop within each expert (e.g., one expert for arithmetic, another for logical deduction).
    * **Modular Reasoning:** Encapsulates deep, localized reasoning within distinct, modular units.
* **Drawbacks:**
    * **Extreme Computational Cost:** The `k*r` cost multiplier makes this approach very slow and resource-intensive.
    * **High Parameter Count:** Storing N independent recurrent blocks is memory-intensive.
    * **Training Complexity:** High risk of redundancy or under-utilization if experts learn similar functions.

#### Option B: Shared Recurrent Block with Projections
* **Architecture & Layer Order:** To address the cost of Option A, this design uses a single recurrent block, `R`, whose parameters are **shared** across all N experts. Each expert, `E_i`, is composed of a unique pair of input and output linear projection layers that "wrap" the shared block. This represents a **"shallow specialization"** model.
* **Token Flow & Data Shapes:** A token `x` is routed to expert `i`. It is first transformed by a unique input projection (`Proj_in_i`), processed for `r` steps by the shared recurrent block `R`, and then transformed by a unique output projection (`Proj_out_i`). The flow is: `y = Proj_out_i(R(Proj_in_i(x)))`.
* **Parameter & FLOP Profile:** This is a highly parameter-efficient compromise. The model only stores one recurrent block per MoE layer, plus the much smaller projection layers for each expert. FLOPs are still multiplied by `r`, but the parameter savings are substantial.
* **Strengths:**
    * **Parameter Efficiency:** Dramatically reduces memory footprint compared to independent experts.
    * **Scalable Specialization:** Expertise is learned in the lightweight projections, which guide the powerful, general-purpose reasoning engine (`R`).
    * **Continual Learning Potential:** The core block `R` can be frozen, and new experts (new projection pairs) can be added and trained on new domains, integrating PEFT principles directly into the architecture to prevent catastrophic forgetting.
* **Drawbacks:**
    * **Generic Core Logic:** The core reasoning process is universal. The model's ability to specialize is limited by the expressiveness of the linear projections.
    * **Potential Bottleneck:** The shared recurrent block could become a bottleneck if the projection layers are not sufficient to adapt inputs for diverse tasks.

### Synthesis II: Recurrence at the MoE Layer Level

This architecture elevates the scope of recurrence, applying it to the entire Mixture of Experts layer, transforming it into a dynamic, iterative processing unit.

#### Option C: The MoE Layer as the Recurrent Unit
* **Architecture & Layer Order:** This is the most direct fusion of the two paradigms. The *entire MoE layer*—including its gating network and the full set of N experts—is treated as the single recurrent block. The output of the layer is fed back as input for the next iteration, typically with a residual connection.
* **Token Flow & Data Shapes:** A hidden state `s_i` is processed by the MoE layer. The output is used to update the state: `s_{i+1} = s_i + MoE(s_i)`. This updated state `s_{i+1}` is then fed back into the *same* MoE layer for the next of `r` total steps.
* **Parameter & FLOP Profile:** The parameter count is identical to a standard MoE model. The cost is purely computational: the FLOPs of the entire MoE layer are multiplied by the number of recurrence steps `r`. This makes the architecture parameter-efficient but inference-slow.
* **Strengths:**
    * **Stateful, Dynamic Routing:** The router's decision at step `i+1` is based on the output of the experts from step `i`. This creates a feedback loop.
    * **Latent Chain-of-Thought:** Enables a multi-step reasoning process where expert selection evolves as the model's understanding deepens with each iteration.
    * **Parameter Simplicity:** Requires no new parameters compared to a standard MoE baseline.
* **Drawbacks:**
    * **Slow Inference:** Repeating the entire MoE layer computation `r` times significantly reduces throughput.
    * **Stability Risk:** The dynamic feedback loop could be unstable, leading to unproductive oscillations or chaotic routing behavior that is difficult to train.
    * **Vanishing/Exploding Gradients:** Backpropagating through many layer-level iterations poses a significant optimization challenge.

---

### Comparative Summary

| Model                          | Param Efficiency | Active Compute (per token) | Specialization Mode           | Continual Adaptation | **Architecture Depth** |
| ------------------------------ | ---------------- | -------------------------- | ----------------------------- | -------------------- | ---------------------- |
| Standard MoE                   | Baseline         | 1× FFN (per layer)         | FFN per expert                | Add experts/FFNs     | **6 separate layers** |
| Recurrent Experts              | Low              | ~`k × r` recurrent steps   | Deep (per-expert recurrent)   | Costly (new `R_i`)   | **1 layer × r recurrence** |
| Shared-Block Experts           | High             | ~`k × r` (shared `R`)      | Shallow (via projections)     | Cheap (new proj pair) | **1 layer × r recurrence** |
| MoE as Recurrent Unit (layer)  | Medium           | ~`r_layer ×` MoE pass      | Stateful routing across iters | Moderate             | **1 layer × r recurrence** |

**Critical Architecture Difference**: 
- **Standard MoE**: Multiple separate transformer layers (N layers)
- **MoRE Variants**: Single transformer layer where the **recurrent block gets repeated r times** to create depth
- **Depth Creation**: Recurrence creates computational depth within a single layer rather than stacking multiple layers

Notes: `k`=top-k experts per token (usually 1); `r`=expert recurrence steps; `r_layer`=MoE-layer iterations. Throughput depends on `r` and implementation (flash attention, checkpointing).

---

## 3. Experimental Pipeline and Timeline

The validation will be conducted on a single **NVIDIA RTX 4070 Super (12GB VRAM)** with 32GB of system RAM. Each of the four models (one baseline, three MoRE variants: Recurrent Experts, Shared-Block Experts, MoE-as-Recurrent-Layer) will be trained for approximately **2-3 days**.

### Pipeline for each model:
1.  **Implementation**: Implement the specified architecture using PyTorch and the DeepSpeed library.
2.  **Pre-training**: Train the model from scratch on the **WikiText-103** dataset for general language modeling. The primary objective is next-token prediction. This phase will last for the majority of the 2-3 day budget.
3.  **Monitoring**: Throughout pre-training, log key metrics (loss, perplexity, expert utilization) to Weights & Biases or TensorBoard to monitor for instability or underperformance.
4.  **Evaluation**:
    *   Periodically evaluate validation perplexity on a held-out split of WikiText-103.
    *   After pre-training (or using checkpoints), evaluate the model's reasoning capabilities on the **bAbI** and **HellaSwag** benchmarks. This can be done via zero-shot/few-shot prompting or a brief fine-tuning phase.
5.  **Analysis**: Compare the results across all five models, focusing on the trade-offs between language modeling performance (perplexity), reasoning ability (accuracy), and computational cost (throughput).

---

## 4. Datasets for Training and Evaluation

To provide a comprehensive assessment, we will use a combination of datasets targeting different capabilities.

*   **General Language Modeling – WikiText-103**: This dataset contains over 100 million tokens from high-quality Wikipedia articles. It is a standard benchmark for measuring a model's ability to capture long-range dependencies and general linguistic patterns. We will use it for pre-training and measure validation perplexity to assess core language modeling performance.

*   **Synthetic Reasoning – Facebook bAbI Tasks**: This suite of 20 question-answering tasks is designed to test specific reasoning skills in a controlled environment. Tasks like "Two-Supporting Facts" and "Three-Supporting Facts" directly probe a model's ability to perform multi-hop reasoning. Success on these tasks would be strong evidence that recurrent experts are enhancing the model's reasoning capacity.

*   **Commonsense Reasoning – HellaSwag**: This benchmark tests a model's ability to choose the most plausible continuation of a short story. It requires contextual understanding and commonsense inference beyond simple pattern matching. We will use HellaSwag to evaluate whether the architectural changes lead to a more robust and common-sense understanding of language.

---

## 5. Architecture Validation: Indicators and Trends

We will systematically monitor training dynamics and evaluation metrics to validate each architecture.

### 5.1 Key Metrics to Track
*   **Training & Validation Loss/Perplexity:** The primary indicator of learning. We will compare the convergence curves of all models.
*   **Downstream Task Accuracy:** Accuracy on bAbI and HellaSwag will measure reasoning and commonsense capabilities.
*   **Expert Load Balance:** The fraction of tokens routed to each expert. A balanced load is desired. The **Coefficient of Variation (CV)** of expert loads will be tracked as a quantitative measure of balance.
*   **Router Gating Entropy:** Measures the "uncertainty" of the router. Very low entropy indicates potential expert collapse.
*   **Auxiliary Load Balancing Loss:** This loss term should decrease over time if the router is successfully balancing the load across experts.

### 5.2 Signs of Success
*   **Improved Performance:** A MoRE variant achieves lower perplexity or higher reasoning accuracy than the standard MoE baseline.
*   **Stable Training:** The model trains without significant loss spikes or gradient explosions.
*   **Healthy Expert Utilization:** All experts receive a reasonable share of tokens throughout training; no expert is starved.
*   **Effective Refinement (for recurrent variants):** The recurrent blocks demonstrably alter the token representations in a beneficial way (e.g., leading to a lower final loss).

### 5.3 Warning Signs of Underperformance
*   **Expert Collapse:** One or a few experts handle the vast majority of tokens, while others are idle. This negates the benefit of MoE.
*   **Training Instability:** The training loss is erratic, spikes frequently, or diverges (returns `NaN`). This is a particular risk for recurrent architectures.
*   **Stagnation:** The model's performance plateaus at a level worse than or equal to the baseline, indicating the architectural changes provided no benefit.
*   **Projection Collapse (Synthesis I — Shared-Block):** Experts' input/output projections converge to similar transforms, reducing specialization.
*   **Unstable Iterative Routing (Synthesis II):** Router oscillates across iterations without converging to useful expert sequences (low entropy but poor accuracy).

---

## 6. Technology Stack

The project will be built using the following open-source tools and frameworks:

*   **Core Framework:** **PyTorch** for model implementation and training loops.
*   **Optimized & Parallel Training:** **DeepSpeed** will be used to:
    *   Enable **mixed-precision training (fp16)** for speed and memory efficiency.
    *   Implement **ZeRO Stage 2 with CPU Offload** to manage memory for large models on a single GPU.
    *   Utilize DeepSpeed's built-in `MoE` layer for efficient and stable routing, including the auxiliary load balancing loss.
*   **Monitoring and Visualization:** **Weights & Biases (W&B)** or **TensorBoard** for real-time logging and comparison of training runs.
*   **Foundation Models & Components:** **Hugging Face Transformers** for tokenizers, model components, and evaluation scripts where applicable.

---

## 7. Technical Implementation Details: The Recurrent Reasoning Block

The recurrent block used in our MoRE variants will be based on the design from the **Huginn** model, as detailed in its implementation (`raven_modeling_minimal.py` available at [https://ollama.hf-mirror.com/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py](https://ollama.hf-mirror.com/tomg-group-umd/huginn-0125/blob/main/raven_modeling_minimal.py)) and the associated paper, "Scaling up Test-Time Compute with Latent Reasoning." This block is essentially a Transformer decoder block with tied weights, applied iteratively.

### Core Components

1.  **Input Adapter**: At each recurrent step, the previous latent state `x` is combined with the original, static token embedding `input_embeds`. This is done by concatenation followed by a linear projection, ensuring the model can continuously reference the original input.

    ```python
    # Simplified from Huginn's core_block_forward
    # 'x' is the latent state from the previous step
    # 'input_embeds' is the static embedding from the prelude
    x = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))
    ```

2.  **SandwichBlock**: The core of the recurrence is a `SandwichBlock`, which contains a standard Transformer sub-layer structure but with specific design choices.

    ```python
    # From raven_modeling_minimal.py
    class SandwichBlock(nn.Module):
        def __init__(self, config, layer_id):
            super().__init__()
            self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.attn   = CausalSelfAttention(config)
            self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.mlp    = GatedMLP(config)
            self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.layer_id = layer_id
            
        def forward(self, x, freqs_cis, step_idx, mask=None, past_key_values=None, return_attn=False):
            attn_out, attn_map = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values, return_attn)
            x = self.norm_2(attn_out + x)
            x = self.norm_4(self.mlp(self.norm_3(x)) + x)
            return x, attn_map
    ```
    Key features include:
    *   **RMSNorm:** A simpler, more efficient normalization layer.
    *   **Pre-LN style:** Normalization is applied before the attention and MLP sub-layers.
    *   **CausalSelfAttention:** Allows interaction between the latent states of tokens within the recurrent step. For MoRE, this attention would be limited to the tokens routed to a specific expert.
    *   **`step_idx`:** A unique index for each recurrent step, ensuring causality in the "depth" dimension and correct positional encoding.

3.  **Gated MLP (gMLP)**: Instead of a standard FFN with ReLU, Huginn uses a **Gated Linear Unit** (specifically SwiGLU), which has shown improved performance in many modern LLMs.

    ```python
    # From raven_modeling_minimal.py
    class GatedMLP(nn.Module):
        def __init__(self, config, in_features=0):
            super().__init__()
            in_features = config.n_embd if in_features == 0 else in_features
            self.fc   = nn.Linear(in_features, config.intermediate_size * 2, bias=False)
            self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
            self.nonlin = nn.SiLU()
            
        def forward(self, x: Tensor) -> Tensor:
            # One FC, then split for gated activation
            x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
            x = self.nonlin(x_fc_1) * x_fc_2
            return self.proj(x)
    ```

4.  **Weight Tying and Iteration Loop**: The same `SandwichBlock` weights are reused across all recurrent steps. The training loop cleverly uses `torch.no_grad()` for some initial steps and only enables gradients for the last few steps. This **truncated backpropagation through time** makes training stable and memory-efficient, even with many recurrent iterations.

    ```python
    # Simplified from Huginn's iterate_forward
    # num_steps_no_grad and num_steps_with_grad are sampled randomly
    
    with torch.no_grad():
        for step in range(num_steps_no_grad):
            # ... call core_block_forward ...
            
    for step in range(num_steps_with_grad):
        # ... call core_block_forward ...
    ```

By integrating this well-tested and powerful recurrent block into our MoE framework, we can explore the full potential of the MoRE architecture. Each of the proposed Synthesis I/II variants leverages these components in distinct structural configurations, enabling a systematic study of their trade-offs.
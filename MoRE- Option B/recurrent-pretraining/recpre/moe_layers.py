import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SwiGLUFFN(nn.Module):
    """
    A standard SwiGLU Feed-Forward Network following the specification.
    
    Structure: input_proj -> SwiGLU -> output_proj
    SwiGLU(x) = (x @ W_gate) * SiLU(x @ W_up)
    """
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation of SwiGLU activation as specified
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(F.silu(gate) * up)


class SharedExpertMoE(nn.Module):
    """
    Shared-Expert Mixture of Experts (S-MoE) layer.
    
    This module contains:
    - 1 shared expert that all tokens pass through
    - 8 specialized experts with top-2 routing
    - Load balancing loss to prevent router collapse
    
    Follows the exact specification from moe_huginn_spec.md
    """
    def __init__(self, d_model: int, d_ffn: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_experts = n_experts
        self.top_k = top_k

        # 1 shared expert that all tokens pass through
        self.shared_expert = SwiGLUFFN(d_model, d_ffn)

        # 8 specialized experts
        self.specialized_experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ffn) for _ in range(n_experts)
        ])

        # Router: single learnable linear layer
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Store metrics for monitoring
        self._last_metrics = None
        
        # Initialize router weights properly
        self._init_router_weights()

    def _init_router_weights(self):
        """Initialize router weights to prevent early router collapse."""
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass implementing the exact specification with a vectorized router.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            tuple: (output, auxiliary_load_balancing_loss)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Shared Path: All tokens pass through shared expert
        shared_output = self.shared_expert(x)
        
        # 2. Sparse Path: Router selects top-2 experts per token
        x_flat = x.view(-1, d_model)
        router_logits = self.router(x_flat)

        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(x.dtype)

        # 3. Compute sparse expert output (vectorized)
        sparse_output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts)

        for i in range(self.n_experts):
            # Find tokens routed to this expert
            batch_idx, k_idx = torch.where(expert_mask[:, :, i])

            if batch_idx.numel() > 0:
                # Get inputs and weights for this expert
                expert_inputs = x_flat[batch_idx]
                expert_weights = routing_weights[batch_idx, k_idx].unsqueeze(1)
                
                # Apply expert and add weighted output to the final result
                expert_outputs = self.specialized_experts[i](expert_inputs)
                sparse_output.index_add_(0, batch_idx, expert_outputs * expert_weights)

        sparse_output = sparse_output.view(batch_size, seq_len, d_model)
        
        # 4. Combine: shared_output + sparse_output
        output = shared_output + sparse_output
        
        # 5. Calculate auxiliary load balancing loss
        aux_loss = self._compute_load_balancing_loss(router_logits, selected_experts, batch_size, seq_len)
        
        # 6. Store comprehensive metrics for monitoring
        self._store_metrics(routing_weights, selected_experts, aux_loss, batch_size, seq_len)
        
        return output, aux_loss

    def _compute_load_balancing_loss(self, router_logits: torch.Tensor, 
                                   selected_experts: torch.Tensor, 
                                   batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss as in Switch Transformer paper.
        
        This encourages an even distribution of tokens across all experts.
        loss = n_experts * sum(f_i * P_i)
        f_i: fraction of tokens dispatched to expert i
        P_i: average router probability for expert i over all tokens
        """
        num_tokens = batch_size * seq_len
        
        # Get router probabilities for all experts
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Calculate f_i: fraction of tokens dispatched to each expert.
        expert_counts = torch.zeros(self.n_experts, device=router_logits.device)
        expert_counts.index_add_(0, selected_experts.view(-1), torch.ones_like(selected_experts.view(-1), dtype=torch.float))
        tokens_per_expert_fraction = expert_counts / num_tokens
        
        # Calculate P_i: average router probability for each expert.
        mean_router_prob_per_expert = router_probs.mean(dim=0)
        
        # The loss is the dot product of these two, scaled by the number of experts.
        loss_aux = self.n_experts * torch.sum(tokens_per_expert_fraction * mean_router_prob_per_expert)
        
        return loss_aux

    def _store_metrics(self, routing_weights: torch.Tensor, selected_experts: torch.Tensor, 
                       aux_loss: torch.Tensor, batch_size: int, seq_len: int):
        """Store comprehensive metrics for monitoring and debugging."""
        with torch.no_grad():
            # Reshape for processing
            flat_weights = routing_weights.reshape(-1)
            flat_experts = selected_experts.reshape(-1)
            
            # Basic metrics
            expert_counts = torch.bincount(flat_experts, minlength=self.n_experts)
            expert_utilization = (expert_counts > 0).float().mean()
            
            # Routing entropy (diversity measure)
            routing_entropy = -(routing_weights * routing_weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
            
            # Expert preference analysis
            expert_preference = torch.zeros(self.n_experts, device=routing_weights.device, dtype=routing_weights.dtype)
            expert_preference.scatter_add_(0, flat_experts, flat_weights)
            expert_preference = expert_preference / (batch_size * seq_len)
            
            # Router collapse detection
            max_expert_attention = expert_preference.max()
            min_expert_attention = expert_preference.min()
            attention_spread = max_expert_attention - min_expert_attention
            
            # Gini coefficient for inequality measurement (FIXED)
            sorted_prefs = torch.sort(expert_preference)[0]
            n = self.n_experts
            if sorted_prefs.sum() > 0:  # Avoid division by zero
                # Correct Gini coefficient formula: G = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
                weighted_sum = torch.arange(1, n + 1, device=sorted_prefs.device, dtype=sorted_prefs.dtype) * sorted_prefs
                gini_coefficient = (2 * weighted_sum.sum() - (n + 1) * sorted_prefs.sum()) / (n * sorted_prefs.sum())
                gini_coefficient = torch.clamp(gini_coefficient, 0.0, 1.0)  # Ensure bounds [0, 1]
            else:
                gini_coefficient = torch.tensor(0.0, device=sorted_prefs.device, dtype=sorted_prefs.dtype)
            
            # Router collapse warnings
            top_expert_usage_pct = max_expert_attention * 100
            router_collapse_warning = top_expert_usage_pct > 50.0
            low_expert_diversity_warning = expert_utilization < 0.5
            
            # Per-expert detailed metrics
            per_expert_metrics = {}
            for i in range(self.n_experts):
                per_expert_metrics[f"expert_{i}_token_ratio"] = expert_counts[i] / (batch_size * seq_len)
                per_expert_metrics[f"expert_{i}_importance"] = expert_preference[i]
            
            # Store all metrics
            self._last_metrics = {
                "expert_counts": expert_counts.detach(),
                "expert_utilization": expert_utilization.detach(),
                "routing_entropy": routing_entropy.detach(),
                "load_balance_loss": aux_loss,
                "expert_preference": expert_preference.detach(),
                "max_expert_attention": max_expert_attention.detach(),
                "min_expert_attention": min_expert_attention.detach(),
                "attention_spread": attention_spread.detach(),
                "gini_coefficient": gini_coefficient.detach(),
                "top_expert_usage_pct": top_expert_usage_pct.detach(),
                "router_collapse_warning": router_collapse_warning,
                "low_expert_diversity_warning": low_expert_diversity_warning,
                "attention_variance": expert_preference.var().detach(),
                **per_expert_metrics
            }

    def get_last_metrics(self) -> Optional[dict]:
        """Return the metrics from the last forward pass."""
        return self._last_metrics


# Keep existing implementations for backward compatibility
class MoERouter(nn.Module):
    """Top-k MoE router for expert selection."""
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning routing weights and selected experts."""
        logits = self.router(x)
        weights, experts = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, experts


class MoELayer(nn.Module):
    """Standard MoE layer for backward compatibility."""
    def __init__(self, d_model: int, num_experts: int, k: int = 2, expansion_factor: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.d_ffn = int(d_model * expansion_factor)
        
        # Create experts
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, self.d_ffn) for _ in range(num_experts)
        ])
        
        # Router
        self.router = MoERouter(d_model, num_experts, k)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer."""
        batch_size, seq_len, d_model = x.shape
        
        # Get routing weights and selected experts
        routing_weights, selected_experts = self.router(x)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for k_idx in range(self.k):
            expert_indices = selected_experts[:, :, k_idx]
            weights = routing_weights[:, :, k_idx: k_idx + 1]
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx).unsqueeze(-1)
                if mask.any():
                    expert_output = self.experts[expert_idx](x)
                    output += mask * weights * expert_output
        
        return self.norm(output)



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- Core Building Blocks for Models ---

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    As described in the Huginn/Gemini plan.
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class GatedMLP(nn.Module):
    """

    Gated MLP (SwiGLU).
    As described in the Huginn/Gemini plan.
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w_gate(x)))

class CausalSelfAttention(nn.Module):
    """
    Standard Causal Self-Attention.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, t, c = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = rearrange(output, 'b h t d -> b t (h d)')
        return self.out_proj(output)

class RecurrentBlock(nn.Module):
    """
    The Huginn-style recurrent block (SandwichBlock).
    This is the core of MoRE Option A.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, dropout=dropout)
        
        # Adapter to merge latent state with original input
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x, num_recurrences=1):
        """
        x: The input tensor for the expert.
        num_recurrences: The number of times to apply the recurrent loop.
        """
        static_input = x.clone()
        latent_state = torch.randn_like(x) * 0.02 # Initialize latent state

        for _ in range(num_recurrences):
            # 1. Adapt: merge latent state with original input
            merged = torch.cat([latent_state, static_input], dim=-1)
            adapted_state = self.adapter(merged)

            # --- Applying a standard Pre-LN Transformer Block ---
            
            # 2. Self-Attention Block
            # Unsqueeze to add a batch dimension for attention, then squeeze back
            attn_input = self.norm1(adapted_state).unsqueeze(0)
            attn_out = self.attn(attn_input).squeeze(0)
            # First residual connection
            latent_state = adapted_state + attn_out
            
            # 3. Feed-Forward Block
            mlp_input = self.norm2(latent_state)
            mlp_out = self.mlp(mlp_input)
            # Second residual connection
            latent_state = latent_state + mlp_out

        return latent_state

class FeedForward(nn.Module):
    """
    A standard 2-layer Feed-Forward Network.
    Used in the baseline MoE and Options C & D.
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))

class Router(nn.Module):
    """
    The expert router.
    It computes routing weights and handles token dispatching.
    We will use a simplified version that returns indices and combines outputs later.
    """
    def __init__(self, d_model, num_experts, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            - final_output: The combined output from the experts.
            - aux_loss: The load balancing loss.
        """
        # Reshape for routing: [batch_size * seq_len, d_model]
        x_flat = x.view(-1, self.d_model)
        
        # Get logits: [batch_size * seq_len, num_experts]
        router_logits = self.layer(x_flat)

        # Get routing weights and selected experts
        routing_weights = F.softmax(router_logits, dim=1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        
        # Normalize top_k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- Gating and Load Balancing ---
        # For general top-k, count assignments per expert across all tokens
        # one_hot over experts then sum over k -> [T, E]
        assignment_matrix = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1).float()
        # tokens_per_expert counts total assignments (T*K when summed)
        tokens_per_expert = assignment_matrix.sum(dim=0)  # [E]
        # Importance per expert from soft routing probabilities (sum over tokens)
        importance_per_expert = routing_weights.sum(dim=0)  # [E]
        
        # Normalize to fractions
        total_assignments = float(top_k_indices.numel())  # T * K
        total_tokens = float(x_flat.size(0))              # T
        load = tokens_per_expert / max(1.0, total_assignments)
        importance = importance_per_expert / max(1.0, total_tokens)
        
        # Load balancing loss (Switch-style simplification)
        load_balance_loss = self.num_experts * torch.sum(load * importance)

        return {
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "load_balance_loss": load_balance_loss,
            "router_logits": router_logits  # Pass this up for logging
        } 
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CausalSelfAttention, RMSNorm, Router

class ScaledDownHuginnExpert(nn.Module):
    """
    Synthesis I Option A: Independent Recurrent Experts.
    
    Each expert is a scaled-down version of the Huginn recurrent transformer block.
    This creates a "deep specialization" model where each expert can learn a unique,
    complex internal algorithm through recurrence.
    
    Architecture: Each expert is a complete, independent recurrent transformer block
    with fewer parameters than the full Huginn model.
    """
    def __init__(self, d_model, num_steps=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps
        self.n_heads = n_heads
        
        # Input adapter - merges current state with static input (scaled down)
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False)
        
        # Core transformer-like recurrent block (scaled down from Huginn)
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        
        # Gated MLP (scaled down SwiGLU style)
        self.norm3 = RMSNorm(d_model)
        # Reduced intermediate size for scaling down
        intermediate_size = 2 * d_model  # Instead of 4 * d_model in full Huginn
        self.mlp = nn.Sequential(
            nn.Linear(d_model, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, d_model, bias=False)
        )
        self.norm4 = RMSNorm(d_model)
        
    def forward(self, x, num_steps=None):
        """
        Forward pass through the scaled-down Huginn expert.
        
        Args:
            x: Input tensor [num_tokens_for_expert, d_model]
            num_steps: Number of recurrent steps (overrides config if provided)
        
        Returns:
            output: Final state after recurrence [num_tokens_for_expert, d_model]
        """
        if num_steps is None:
            num_steps = self.num_steps
            
        batch_size = x.size(0)
        device = x.device
        
        # Initialize latent state (random as in Huginn, but scaled)
        state = torch.randn(batch_size, self.d_model, device=device) * 0.02
        static_input = x.clone()  # Keep original input
        
        for step in range(num_steps):
            # Adapt current state with static input
            combined = torch.cat([state, static_input], dim=-1)
            state = self.adapter(combined)
            
            # Self-attention (scaled down)
            residual = state
            state = self.norm1(state)
            
            # For single-token expert, self-attention might be simplified
            # In full implementation, this would attend over sequence
            # CausalSelfAttention expects (x, mask=None) where x contains Q, K, V
            attn_out = self.attn(
                state.unsqueeze(1)  # Add sequence dimension, single argument
            )
            state = attn_out.squeeze(1)  # Remove sequence dimension
            
            state = self.norm2(residual + state)
            
            # Gated MLP (scaled down)
            residual = state
            state = self.norm3(state)
            state = self.mlp(state)
            state = self.norm4(residual + state)
            
        return state

class MoRELayerSynthesisIOptionA(nn.Module):
    """
    MoE Layer with Independent Recurrent Experts (Synthesis I Option A).
    
    This layer contains:
    - Router that selects experts based on input characteristics
    - N independent, scaled-down Huginn recurrent transformer experts
    - Each expert has its own complete set of parameters
    """
    def __init__(self, d_model, num_experts, top_k=1, dropout=0.1, 
                 num_recurrences=2, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_recurrences = num_recurrences
        
        # Router
        self.router = Router(d_model, num_experts, top_k)
        
        # Independent recurrent experts (each is a scaled-down Huginn block)
        self.experts = nn.ModuleList([
            ScaledDownHuginnExpert(d_model, num_recurrences, n_heads, dropout)
            for _ in range(num_experts)
        ])
        
    def forward(self, x, num_recurrences=None):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Get routing decisions
        routing_info = self.router(x)
        top_k_indices = routing_info["top_k_indices"]
        top_k_weights = routing_info["top_k_weights"]
        K = top_k_indices.size(1)
        
        # Expand tokens per assignment
        tokens_expanded = x_flat.unsqueeze(1).expand(-1, K, -1).reshape(-1, d_model)
        assigned_experts = top_k_indices.reshape(-1)
        assigned_weights = top_k_weights.reshape(-1)
        token_ids_expanded = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
        
        # Use provided num_recurrences or default
        if num_recurrences is None:
            num_recurrences = self.num_recurrences
        
        final_output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for i, expert in enumerate(self.experts):
            indices = torch.nonzero(assigned_experts == i, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
                
            expert_input = tokens_expanded[indices]
            expert_output = expert(expert_input, num_steps=num_recurrences)
            weights_for_expert = top_k_weights.view(-1)[indices]
            token_ids_for_expert = token_ids_expanded[indices]
            
            final_output.index_add_(0, token_ids_for_expert, 
                                  expert_output * weights_for_expert.unsqueeze(1))
        
        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output, routing_info["load_balance_loss"], routing_info["router_logits"]

class MoRETransformerBlockSynthesisIOptionA(nn.Module):
    """
    Transformer block for Synthesis I Option A: Independent Recurrent Experts.
    """
    def __init__(self, d_model, n_heads, num_experts, top_k=1, dropout=0.1, 
                 num_recurrences=2):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.moe = MoRELayerSynthesisIOptionA(d_model, num_experts, top_k, dropout,
                                             num_recurrences=num_recurrences)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, num_recurrences=None, mask=None):
        # Self-attention
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        
        # MoE layer with independent recurrent experts
        moe_out, load_balance_loss, router_logits = self.moe(self.norm2(x), num_recurrences=num_recurrences)
        x = x + moe_out
        
        return x, load_balance_loss, router_logits

class MoREModelSynthesisIOptionA(nn.Module):
    """
    The full model for Synthesis I Option A: Independent Recurrent Experts.
    
    This model embodies "deep specialization" where each expert is a complete,
    independent scaled-down Huginn recurrent transformer block.
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12, num_experts=8, 
                 top_k=1, dropout=0.1, num_recurrences=2, aux_loss_weight=0.01):
        super().__init__()
        self.num_recurrences = num_recurrences
        self.aux_loss_weight = aux_loss_weight
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(2048, d_model)
        
        self.layers = nn.ModuleList([
            MoRETransformerBlockSynthesisIOptionA(d_model, n_heads, num_experts, top_k, dropout,
                                                num_recurrences=num_recurrences) 
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos_emb = self.positional_embedding(pos)
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        
        total_aux_loss = 0.0
        all_router_logits = []
        
        for layer in self.layers:
            x, aux_loss, router_logits = layer(x, num_recurrences=self.num_recurrences, mask=mask)
            total_aux_loss += aux_loss
            all_router_logits.append(router_logits)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss += self.aux_loss_weight * (total_aux_loss / len(self.layers))
            
        return {
            "logits": logits, 
            "loss": loss, 
            "aux_loss": total_aux_loss, 
            "router_logits": all_router_logits
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, 
                do_sample=False, pad_token_id=None, eos_token_id=None, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :] / max(1e-6, temperature)

            if do_sample:
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)

            token_val = next_token.item() if next_token.numel() == 1 else None
            if pad_token_id is not None and token_val == pad_token_id:
                break
            if eos_token_id is not None and token_val == eos_token_id:
                break

        return input_ids

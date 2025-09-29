import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CausalSelfAttention, FeedForward, RMSNorm, Router

class MoELayer(nn.Module):
    """
    A standard Mixture of Experts layer.
    It contains a router and a list of feed-forward experts.
    """
    def __init__(self, d_model, num_experts, top_k=1, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model) # [B*T, D]

        # Get routing decisions
        routing_info = self.router(x)
        top_k_indices = routing_info["top_k_indices"] # [B*T, top_k]
        top_k_weights = routing_info["top_k_weights"] # [B*T, top_k]
        
        # Initialize final output
        final_output = torch.zeros_like(x_flat)
        
        # This is a simplified, non-parallel implementation for clarity.
        # In a real system, this would be a highly optimized scatter-gather operation.
        flat_indices = top_k_indices.view(-1)
        
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            token_indices_for_expert = torch.where(flat_indices == i)[0]
            
            if token_indices_for_expert.numel() > 0:
                # Select the tokens for this expert
                expert_input = x_flat[token_indices_for_expert]
                
                # Process through the expert
                expert_output = expert(expert_input)
                
                # Get the corresponding weights
                weights_for_expert = top_k_weights.view(-1)[token_indices_for_expert]
                
                # Weight the output and add to the final result
                final_output.index_add_(0, token_indices_for_expert, expert_output * weights_for_expert.unsqueeze(1))
        
        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output, routing_info["load_balance_loss"], routing_info["router_logits"]

class MoETransformerBlock(nn.Module):
    """
    A full Transformer block with Multi-Head Attention and a MoE layer.
    """
    def __init__(self, d_model, n_heads, num_experts, top_k=1, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.moe = MoELayer(d_model, num_experts, top_k, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        
        moe_out, load_balance_loss, router_logits = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, load_balance_loss, router_logits

class BaseMoEModel(nn.Module):
    """
    The baseline Mixture of Experts model.
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12, num_experts=8, top_k=1, dropout=0.1, aux_loss_weight=0.01):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(2048, d_model) # Max sequence length
        self.aux_loss_weight = aux_loss_weight
        
        self.layers = nn.ModuleList(
            [MoETransformerBlock(d_model, n_heads, num_experts, top_k, dropout) for _ in range(n_layers)]
        )
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
            x, aux_loss, router_logits = layer(x, mask)
            total_aux_loss += aux_loss
            all_router_logits.append(router_logits)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss += self.aux_loss_weight * (total_aux_loss / len(self.layers)) # Weighted and averaged aux loss
            
        return {"logits": logits, "loss": loss, "aux_loss": total_aux_loss, "router_logits": all_router_logits} 
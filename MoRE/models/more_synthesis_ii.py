import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CausalSelfAttention, FeedForward, RMSNorm, Router

class RecurrentMoELayer(nn.Module):
    """
    Synthesis II: MoE Layer as Recurrent Unit.
    
    This architecture treats the *entire MoE layer*—including its gating network 
    and the full set of N experts—as the single recurrent block. The output of 
    the layer is fed back as input for the next iteration, typically with a 
    residual connection.
    
    This enables:
    - Stateful, dynamic routing where router decisions evolve across iterations
    - Latent chain-of-thought reasoning process
    - Expert selection that adapts as understanding deepens
    - The router's decision at step i+1 is based on the output of the experts from step i
    """
    def __init__(self, d_model, num_experts, top_k=1, d_ff=None, dropout=0.1, 
                 num_iters=2, residual_scale=1.0, min_entropy=0.8, logit_ema=0.9):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_iters = num_iters
        self.residual_scale = residual_scale
        self.min_entropy = min_entropy
        self.logit_ema = logit_ema
        
        # Router
        self.router = Router(d_model, num_experts, top_k)
        
        # Standard FFN experts (as in baseline MoE)
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # EMA buffer for router logits to stabilize routing
        self.register_buffer('ema_router_logits', torch.zeros(1, 1, num_experts))
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Initialize state
        state = x_flat
        all_states = []
        all_router_logits = []
        all_expert_usage = []
        
        for iter_step in range(self.num_iters):
            # Get routing decisions based on current state
            # This is the key: router sees the current state and makes decisions
            routing_info = self.router(state.view(batch_size, seq_len, d_model))
            router_logits = routing_info["router_logits"]
            top_k_indices = routing_info["top_k_indices"]
            top_k_weights = routing_info["top_k_weights"]
            
            # Update EMA for router logits (stabilization)
            if self.training:
                if self.ema_router_logits.numel() == 1:  # First iteration
                    self.ema_router_logits = router_logits.clone()
                else:
                    self.ema_router_logits = (
                        self.logit_ema * self.ema_router_logits + 
                        (1 - self.logit_ema) * router_logits
                    )
            
            # Enforce minimum entropy constraint
            router_probs = F.softmax(router_logits, dim=-1)
            entropy = -(router_probs * router_probs.log()).sum(dim=-1).mean()
            
            if entropy < self.min_entropy:
                # Add noise to increase entropy
                noise = torch.randn_like(router_logits) * 0.1
                router_logits = router_logits + noise
                # Recompute routing
                routing_info = self.router(state.view(batch_size, seq_len, d_model))
                top_k_indices = routing_info["top_k_indices"]
                top_k_weights = routing_info["top_k_weights"]
            
            # Process tokens through selected experts
            expert_outputs = torch.zeros_like(state)
            expert_usage = torch.zeros(self.num_experts, device=state.device)
            
            K = top_k_indices.size(1)
            tokens_expanded = state.unsqueeze(1).expand(-1, K, -1).reshape(-1, d_model)
            assigned_experts = top_k_indices.reshape(-1)
            assigned_weights = top_k_weights.reshape(-1)
            token_ids_expanded = torch.arange(state.size(0), device=state.device).unsqueeze(1).expand(-1, K).reshape(-1)
            
            for i, expert in enumerate(self.experts):
                indices = torch.nonzero(assigned_experts == i, as_tuple=False).squeeze(-1)
                if indices.numel() > 0:
                    expert_input = tokens_expanded[indices]
                    expert_output = expert(expert_input)
                    weights_for_expert = assigned_weights[indices]
                    token_ids_for_expert = token_ids_expanded[indices]
                    
                    expert_outputs.index_add_(0, token_ids_for_expert, 
                                            expert_output * weights_for_expert.unsqueeze(1))
                    
                    # Track expert usage
                    expert_usage[i] = indices.numel()
            
            # Normalize expert usage
            total_tokens = state.size(0)
            expert_usage = expert_usage / max(total_tokens, 1.0)
            
            # Update state with residual connection
            # This creates the feedback loop: state_{i+1} = state_i + MoE(state_i)
            delta = expert_outputs
            state = state + self.residual_scale * delta
            
            # Store intermediate information
            all_states.append(state.clone())
            all_router_logits.append(router_logits)
            all_expert_usage.append(expert_usage)
        
        # Final output is the last state
        final_output = state.view(batch_size, seq_len, d_model)
        
        # Compute average load balancing loss across iterations
        avg_load_balance_loss = routing_info["load_balance_loss"]
        
        return final_output, avg_load_balance_loss, all_router_logits, all_expert_usage, all_states

class RecurrentMoETransformerBlock(nn.Module):
    """
    Transformer block for Synthesis II: MoE Layer as Recurrent Unit.
    """
    def __init__(self, d_model, n_heads, num_experts, top_k=1, dropout=0.1, 
                 num_iters=2, residual_scale=1.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.moe = RecurrentMoELayer(d_model, num_experts, top_k, dropout=dropout,
                                   num_iters=num_iters, residual_scale=residual_scale)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        
        # Recurrent MoE layer - the entire MoE layer is reused multiple times
        moe_out, load_balance_loss, router_logits, expert_usage, all_states = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, load_balance_loss, router_logits, expert_usage, all_states

class MoREModelSynthesisII(nn.Module):
    """
    The full model for Synthesis II: MoE Layer as Recurrent Unit.
    
    This model treats the entire MoE layer (router + experts) as a recurrent block
    that gets reused again and again, creating a feedback loop where router
    decisions at step i+1 are based on expert outputs from step i.
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12, num_experts=8, 
                 top_k=1, dropout=0.1, num_iters=2, residual_scale=1.0, aux_loss_weight=0.01):
        super().__init__()
        self.num_iters = num_iters
        self.aux_loss_weight = aux_loss_weight
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(2048, d_model)
        
        self.layers = nn.ModuleList([
            RecurrentMoETransformerBlock(d_model, n_heads, num_experts, top_k, dropout,
                                       num_iters=num_iters, residual_scale=residual_scale) 
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, return_iteration_info=False):
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
        all_expert_usage = []
        all_layer_states = []
        
        for layer in self.layers:
            x, aux_loss, router_logits, expert_usage, layer_states = layer(x, mask)
            total_aux_loss += aux_loss
            all_router_logits.append(router_logits)
            all_expert_usage.append(expert_usage)
            all_layer_states.append(layer_states)
            
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss += self.aux_loss_weight * (total_aux_loss / len(self.layers))
            
        output = {
            "logits": logits, 
            "loss": loss, 
            "aux_loss": total_aux_loss, 
            "router_logits": all_router_logits,
            "expert_usage": all_expert_usage
        }
        
        if return_iteration_info:
            output["layer_states"] = all_layer_states
            
        return output

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


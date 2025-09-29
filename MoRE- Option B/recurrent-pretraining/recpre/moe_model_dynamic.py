"""
MoE-Huginn Model Integration

This module integrates the SharedExpertMoE layer with the existing Huginn architecture,
ensuring that expert-modified inputs are passed through the recurrent block at every iteration.

The key innovation: after every iteration of the recurrent block, the inputs passed along
are the expert-modified inputs, not the original initial inputs.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .model_dynamic import RecurrentGPT
from .moe_layers import SharedExpertMoE


class MoEHuginnModel(RecurrentGPT):
    """
    MoE-Huginn model that extends RecurrentGPT with proper expert-modified input consistency.
    
    This model ensures that after every iteration of the recurrent block,
    the initial inputs getting passed along are the inputs which are modified
    by the Experts, not the actual initial initial inputs.
    
    Key Changes:
    1. Replaces the standard MoE layer with SharedExpertMoE
    2. Modifies the forward pass to store expert-modified inputs
    3. Overrides iterate_forward to use expert-modified inputs in every iteration
    4. Maintains all existing Huginn functionality
    """
    
    def __init__(self, config, **kwargs):
        # Initialize the base RecurrentGPT model
        super().__init__(config, **kwargs)
        
        # Replace the standard MoE layer with our SharedExpertMoE
        if hasattr(self.transformer, 'moe') and self.transformer.moe is not None:
            # Create the new SharedExpertMoE layer
            self.transformer.moe = SharedExpertMoE(
                d_model=config.n_embd,
                d_ffn=getattr(config, 'intermediate_size', config.n_embd * 4),
                n_experts=getattr(config, 'num_experts', 8),
                top_k=getattr(config, 'moe_k', 2)
            )
            
            # Store the expert-modified inputs for use in recurrent iterations
            self._expert_modified_inputs = None
            
            # Flag to track if we're using the new MoE implementation
            self._using_shared_expert_moe = True
        else:
            self._using_shared_expert_moe = False
    
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """
        Forward pass through MoE-Huginn model.
        
        The key difference is that we ensure expert-modified inputs are used
        in every recurrent iteration, not just the first one.
        """
        # Get the original input embeddings
        input_embeds = self.transformer.wte(input_ids)
        if hasattr(self, 'emb_scale') and self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale
            
        # Apply prelude layers
        for _, block in enumerate(self.transformer.prelude):
            input_embeds = block(input_embeds, kwargs.get('freqs_cis'), kwargs.get('attention_mask'))
        
        # Apply MoE layer and get expert-modified inputs
        if self._using_shared_expert_moe:
            # Apply SharedExpertMoE to get expert-modified inputs
            expert_modified_inputs, aux_loss = self.transformer.moe(input_embeds)
            expert_modified_inputs = self.transformer.moe_norm(expert_modified_inputs)
            
            # Store for use in recurrent iterations
            self._expert_modified_inputs = expert_modified_inputs
            
            # Apply MoE normalization
            input_embeds = expert_modified_inputs
        else:
            # Fallback to original MoE implementation
            input_embeds = self.transformer.moe_norm(self.transformer.moe(input_embeds))
            aux_loss = None
        
        # Continue with the standard forward pass, but ensure expert-modified inputs
        # are used in recurrent iterations
        result = super().forward(input_ids, **kwargs)
        
        # Add auxiliary loss if available
        if aux_loss is not None:
            if 'loss' in result and result['loss'] is not None:
                result['loss'] = result['loss'] + aux_loss
            else:
                result['loss'] = aux_loss
                
        return result
    
    def iterate_forward(self, input_embeds, freqs_cis, mask, num_steps_pair=None):
        """
        Override the iterate_forward method to ensure expert-modified inputs
        are used in every recurrent iteration.
        
        This is the critical method that ensures the key requirement:
        "after every iteration of the recurrent block, the initial inputs getting passed along
        are the inputs which are modified by the Experts, not the actual initial initial inputs"
        """
        x = self.initialize_state(input_embeds)
        
        if num_steps_pair is None:
            num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler()
        elif len(num_steps_pair) > 1:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair, torch.tensor(0)
        
        if self.config.randomize_embed_step:
            offset = torch.randint(0, self.config.mean_recurrence * 8, (1,), device=input_embeds.device)
        else:
            offset = 0
        
        # KEY INNOVATION: Use expert-modified inputs if available
        # This ensures that at every iteration, we use the expert-modified inputs
        # not the original initial inputs
        recurrent_inputs = self._expert_modified_inputs if hasattr(self, '_expert_modified_inputs') else input_embeds
        
        with torch.no_grad():
            for step in range(num_steps_no_grad):
                xk = x
                # Pass expert-modified inputs to core block at every iteration
                x = self.core_block_forward(xk, recurrent_inputs, freqs_cis, mask, step + offset)
        
        for step in range(num_steps_with_grad):
            xk = x
            if self.gradient_checkpointing and "per-iteration" in self.config.activation_checkpoint_impl:
                x = self.config.checkpoint(
                    self.core_block_forward, xk, recurrent_inputs, freqs_cis, mask, 
                    num_steps_no_grad + step + offset
                )
            else:
                # Pass expert-modified inputs to core block at every iteration
                x = self.core_block_forward(xk, recurrent_inputs, freqs_cis, mask, 
                                          num_steps_no_grad + step + offset)
        
        return self.transformer.ln_f(x), num_steps_no_grad, num_steps_with_grad, xk.detach()
    
    def core_block_forward(self, x, input_embeds, freqs_cis, mask, step):
        """
        Override core_block_forward to ensure expert-modified inputs are used.
        """
        if self.config.embed_step:
            context = self.step_embedding(torch.as_tensor([step], device=input_embeds.device))
        else:
            context = None
        
        # Use the expert-modified inputs (not original inputs)
        if self.config.injection_type == "add":
            x = x + input_embeds  # This is now expert-modified inputs
        elif self.config.injection_type == "gate":
            x = x * input_embeds  # This is now expert-modified inputs
        elif self.config.injection_type in ["linear", "ffn"]:
            x = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))
        elif self.config.injection_type == "modulated":
            context = x.clone()
        else:
            raise ValueError("Invalid injection type")
        
        # Apply intermediate noise injection if configured
        if self.config.intermediate_noise_injection > 0:
            n = self.config.intermediate_noise_injection
            if self.config.geom_noise_injection == "geom":
                step1 = torch.as_tensor(step + 1, device=x.device)
                x = x * (1 - n / step1) + torch.randn_like(x) * n / step1
            elif self.config.geom_noise_injection == "sqrt":
                step1sqrt = torch.as_tensor(step + 1, device=x.device).sqrt()
                x = x * (1 - n / step1sqrt) + torch.randn_like(x) * n / step1sqrt
            elif self.config.geom_noise_injection == "line":
                noise = max(n, (getattr(self.config, 'maximal_recurrence', 32) - step) / getattr(self.config, 'maximal_recurrence', 32))
                x = x * (1 - noise) + torch.randn_like(x) * noise
            else:
                x = x * (1 - n) + torch.randn_like(x) * n
        
        # Apply core blocks
        if hasattr(self.transformer.core_block[0], 'expanded') and self.transformer.core_block[0].expanded:
            for _, block in enumerate(self.transformer.core_block):
                if not self.gradient_checkpointing:
                    x = block(x, freqs_cis, mask, context=context)
                else:
                    x = self.config.checkpoint(block, x, freqs_cis, mask, context=context)
        else:
            if context is not None:
                x = x + context
            
            for _, block in enumerate(self.transformer.core_block):
                if self.gradient_checkpointing and "per-block" in self.config.activation_checkpoint_impl:
                    x = self.config.checkpoint(block, x, freqs_cis, mask)
                else:
                    x = block(x, freqs_cis, mask)
        
        return x
    
    def get_moe_metrics(self):
        """
        Get MoE metrics for monitoring and logging.
        
        Returns comprehensive metrics including:
        - Load balancing loss
        - Router collapse warnings
        - Expert utilization statistics
        - Routing entropy and diversity measures
        """
        if (self._using_shared_expert_moe and 
            hasattr(self.transformer, 'moe') and 
            hasattr(self.transformer.moe, 'get_last_metrics')):
            return self.transformer.moe.get_last_metrics()
        return None


# Backward compatibility: keep the old class name
MoERecurrentBlock = MoEHuginnModel

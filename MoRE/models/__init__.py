# This file makes the 'models' directory a Python package.

# Import all MoRE model variants
from .moe import BaseMoEModel, MoELayer, MoETransformerBlock
from .more_synthesis_i_option_a import MoREModelSynthesisIOptionA, MoRELayerSynthesisIOptionA, MoRETransformerBlockSynthesisIOptionA
from .more_synthesis_i_option_b import MoREModelSynthesisIOptionB, MoRELayerSynthesisIOptionB, MoRETransformerBlockSynthesisIOptionB
from .more_synthesis_ii import MoREModelSynthesisII, RecurrentMoELayer, RecurrentMoETransformerBlock

# Import core components
from .layers import (
    RMSNorm, GatedMLP, CausalSelfAttention, RecurrentBlock, 
    FeedForward, Router
)

__all__ = [
    # Baseline MoE
    'BaseMoEModel', 'MoELayer', 'MoETransformerBlock',
    
    # Synthesis I: Recurrence at Expert Level
    'MoREModelSynthesisIOptionA', 'MoRELayerSynthesisIOptionA', 'MoRETransformerBlockSynthesisIOptionA',  # Independent Recurrent Experts
    'MoREModelSynthesisIOptionB', 'MoRELayerSynthesisIOptionB', 'MoRETransformerBlockSynthesisIOptionB',  # Shared Recurrent Block with Projections
    
    # Synthesis II: MoE Layer as Recurrent Unit
    'MoREModelSynthesisII', 'RecurrentMoELayer', 'RecurrentMoETransformerBlock',
    
    # Core Components
    'RMSNorm', 'GatedMLP', 'CausalSelfAttention', 'RecurrentBlock', 
    'FeedForward', 'Router'
] 
import sys
import os

# Add the current directory to sys.path so we can import recpre
sys.path.append(os.getcwd())

try:
    from recpre.raven_config_minimal import RavenConfig
    from recpre.raven_modeling_minimal import RavenForCausalLM
    import torch
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback to manual calculation if imports fail (e.g. missing flex_attention)
    print("Falling back to manual calculation based on config values...")
    
    # Hardcoded values from the file we read
    n_embd = 1536
    n_heads = 12
    n_layers = 8 # Total physical layers (prelude + core + coda)
    vocab_size = 65536
    intermediate_size = 14336
    tie_embeddings = True
    
    # Calculate
    embedding_params = vocab_size * n_embd
    
    # Per block params
    # Attention: Wqkv + proj
    # Wqkv: n_embd * (3 * n_embd) (assuming n_kv_heads = n_heads)
    attn_params = n_embd * 3 * n_embd + n_embd * n_embd
    
    # MLP: GatedMLP
    # fc: n_embd * (intermediate_size * 2)
    # proj: intermediate_size * n_embd
    mlp_params = n_embd * intermediate_size * 2 + intermediate_size * n_embd
    
    # Norms: 4 per block * n_embd
    norm_params = 4 * n_embd
    
    block_params = attn_params + mlp_params + norm_params
    
    total_params = embedding_params + n_layers * block_params
    
    # Head (if tied, 0 extra params, otherwise vocab * n_embd)
    if not tie_embeddings:
        total_params += vocab_size * n_embd
        
    print(f"Calculated Parameters (Manual): {total_params:,}")
    print(f"Embeddings: {embedding_params:,}")
    print(f"Per Block: {block_params:,}")
    sys.exit(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    config = RavenConfig()
    print(f"Config loaded: {config}")
    
    # We might not be able to instantiate the full model if dependencies are missing
    # But we can try.
    try:
        # Mock flex_attention if needed or just try instantiation
        model = RavenForCausalLM(config)
        total_params = count_parameters(model)
        print(f"Total Parameters (From Model): {total_params:,}")
        
        # Breakdown
        emb_params = count_parameters(model.transformer.wte)
        print(f"Embeddings: {emb_params:,}")
        
        # Check one block
        if len(model.transformer.core_block) > 0:
            block_params = count_parameters(model.transformer.core_block[0])
            print(f"Core Block Params: {block_params:,}")
            
    except Exception as e:
        print(f"Model instantiation failed: {e}")
        # Fallback to manual calc using config object values
        n_embd = config.n_embd
        n_layers = config.n_layers_in_prelude + config.n_layers_in_recurrent_block + config.n_layers_in_coda
        vocab_size = config.vocab_size
        intermediate_size = config.intermediate_size
        
        embedding_params = vocab_size * n_embd
        attn_params = n_embd * 3 * n_embd + n_embd * n_embd
        mlp_params = n_embd * intermediate_size * 2 + intermediate_size * n_embd
        norm_params = 4 * n_embd
        
        block_params = attn_params + mlp_params + norm_params
        total_params = embedding_params + n_layers * block_params
        
        print(f"Calculated Parameters (From Config Object): {total_params:,}")

if __name__ == "__main__":
    main()

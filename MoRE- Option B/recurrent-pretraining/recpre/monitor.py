"""Glue functions to enable/disable monitoring."""

import torch
from torch.utils.flop_counter import FlopCounterMode


import copy
import time


def enable_monitoring_on_step(module):
    if hasattr(module, "monitoring"):
        module.monitoring = True


def disable_monitoring_and_retrieve_metrics(module, metrics):
    if hasattr(module, "monitoring"):
        if len(module.latest_metrics) > 0:
            metrics |= module.latest_metrics
            module.latest_metrics = {}
        module.monitoring = False


@torch.compile()
def _reverse_engineer_adam_effective_lr(param, param_state, group):
    """Recompute select values to compute Adam effective lr. Ignoring bias correction because ...
    Ignoring actual lr, because its schedule otherwise dominates the effect. Plot the 'actual lr' as this value multiplied by the schedule
    """
    exp_avg = param_state["exp_avg"].float()
    denom = param_state["exp_avg_sq"].float().sqrt().add_(group["eps"])
    effective_lr = torch.where(
        param.grad.float().abs() > group["eps"],
        exp_avg / denom / param.grad.float(),
        exp_avg / denom / group["eps"],  # needs a more stable impl for small values
    )
    return effective_lr


@torch.no_grad()
def track_gradient_metrics(model, optimizer, metrics):
    optim_metrics = {}
    _, H = model.config.padded_vocab_size, model.config.n_embd
    dim_Q, dim_KV = H, model.config.head_size * model.config.n_query_groups
    # Get specific gradient norms:
    qkv_layer_counter, mlp_layer_counter = 0, 0
    qkv_param_ids = []  # store references to check against optim params in 2nd loop
    proj_param_ids = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "qkv" in name and "weight" in name:
                qkv_param_ids += [param]
                if (~torch.isfinite(param.grad)).sum() == 0:
                    if param.grad.numel() % H == 0:  # need to find a better solution for sharded grads
                        q_grad = param.grad.view(-1, H)[:dim_Q, :]
                        optim_metrics[f"query_grad_{qkv_layer_counter}"] = q_grad.norm()
                else:
                    optim_metrics[f"query_grad_{qkv_layer_counter}"] = torch.as_tensor(float("NaN"))
                qkv_layer_counter += 1
            if "mlp" in name and "proj" in name and "weight" in name:
                proj_param_ids += [param]
                if (~torch.isfinite(param.grad)).sum() == 0:
                    optim_metrics[f"ffn2_grad_{mlp_layer_counter}"] = param.grad.norm()
                else:
                    optim_metrics[f"ffn2_grad_{mlp_layer_counter}"] = torch.as_tensor(float("NaN"))
                mlp_layer_counter += 1

    # 2nd moment quality and qkv learning rate
    total_rms = 0.0
    num_params_with_grad = 0
    qkv_layer_counter, mlp_layer_counter = 0, 0
    params_with_finite_grad = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and (~torch.isfinite(param.grad)).sum() == 0:
                params_with_finite_grad.append(param)
                if param in optimizer.state and "exp_avg_sq" in optimizer.state[param]:
                    exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                    if exp_avg_sq.shape == param.grad.shape:
                        rms = (
                            param.grad.float()
                            .pow(2)
                            .div_(exp_avg_sq.float().clamp_(min=group["eps"] ** 2))
                            .mean()
                            .sqrt()
                        )
                        total_rms += rms
                        num_params_with_grad += 1
                        if param is model.transformer.wte.weight:
                            optim_metrics["embed_RMS"] = rms

                        if any(param is p for p in qkv_param_ids):  # comprehension because "in" breaks
                            qkv_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            if qkv_lr.numel() % H == 0:  # need to find a better solution for sharded grads
                                optim_metrics[f"q_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    :dim_Q, :
                                ].mean()
                                optim_metrics[f"k_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    dim_Q : dim_Q + dim_KV, :
                                ].mean()
                                optim_metrics[f"v_effective_lr_{qkv_layer_counter}"] = qkv_lr.view(-1, H)[
                                    dim_Q + dim_KV :, :
                                ].mean()
                            qkv_layer_counter += 1

                        if any(param is p for p in proj_param_ids):
                            proj_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            optim_metrics[f"ffn2_effective_lr_{mlp_layer_counter}"] = proj_lr.mean()
                            mlp_layer_counter += 1

    if num_params_with_grad > 0:
        optim_metrics["avg_RMS"] = total_rms / num_params_with_grad

    # Finally, a quick local L1 norm
    if len(params_with_finite_grad) > 0:
        l1_grad_norm = torch.mean(torch.stack([torch.norm(p.grad.detach(), 1.0) for p in params_with_finite_grad]))
        optim_metrics["local_l1_grad_norm"] = l1_grad_norm

    # And param norms
    l2_param_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
    l1_param_norm = torch.mean(torch.stack([torch.norm(p.detach(), 1.0) for p in model.parameters()]))

    optim_metrics["l2_param_norm"] = l2_param_norm
    optim_metrics["l1_param_norm"] = l1_param_norm

    # sub-group param norms
    optim_metrics["core_block_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for p in model.transformer.core_block.parameters()])
    )
    optim_metrics["word_embed_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for p in model.transformer.wte.parameters()])
    )
    optim_metrics["model_l2_param_norm"] = torch.norm(
        torch.stack([torch.norm(p.detach()) for n, p in model.named_parameters() if "wte" not in name])
    )

    # finalize if all metrics were recorded successfully
    metrics |= optim_metrics


def track_moe_metrics(model: torch.nn.Module, metrics: dict, step: int) -> None:
    """
    Track comprehensive MoE metrics for monitoring router collapse and expert preference.
    
    Args:
        model: The model to monitor
        metrics: Dictionary to store metrics
        step: Current training step
    """
    try:
        # Get MoE metrics from the model
        moe_metrics = None
        if hasattr(model, 'get_moe_metrics'):
            moe_metrics = model.get_moe_metrics()
        elif hasattr(model, 'module') and hasattr(model.module, 'get_moe_metrics'):
            moe_metrics = model.module.get_moe_metrics()
        
        if moe_metrics is None:
            return
        
        # Basic MoE metrics
        if "load_balance_loss" in moe_metrics:
            metrics[f"moe/load_balance_loss"] = moe_metrics["load_balance_loss"].item()
        
        if "routing_entropy" in moe_metrics:
            metrics[f"moe/routing_entropy"] = moe_metrics["routing_entropy"].item()
        
        if "expert_utilization" in moe_metrics:
            metrics[f"moe/expert_utilization"] = moe_metrics["expert_utilization"].item()
        
        # Expert preference analysis
        if "expert_preference" in moe_metrics:
            expert_pref = moe_metrics["expert_preference"]
            metrics[f"moe/max_expert_attention"] = expert_pref.max().item()
            metrics[f"moe/min_expert_attention"] = expert_pref.min().item()
            metrics[f"moe/attention_spread"] = (expert_pref.max() - expert_pref.min()).item()
            
            # Calculate attention variance (higher variance indicates more specialization)
            attention_variance = torch.var(expert_pref).item()
            metrics[f"moe/attention_variance"] = attention_variance
        
        # Router collapse detection
        if "expert_counts" in moe_metrics:
            expert_counts = moe_metrics["expert_counts"]
            total_tokens = expert_counts.sum().item()
            
            if total_tokens > 0:
                # Gini coefficient for expert utilization inequality
                sorted_counts = torch.sort(expert_counts.float())[0]
                cumsum = torch.cumsum(sorted_counts, 0)
                n = expert_counts.shape[0]
                gini_coefficient = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
                metrics[f"moe/gini_coefficient"] = gini_coefficient.item()
                
                # Top and bottom expert usage percentages
                top_expert_usage = expert_counts.max().item() / total_tokens
                bottom_expert_usage = expert_counts.min().item() / total_tokens
                metrics[f"moe/top_expert_usage_pct"] = top_expert_usage * 100
                metrics[f"moe/bottom_expert_usage_pct"] = bottom_expert_usage * 100
                
                # Expert diversity
                active_experts = (expert_counts > 0).sum().item()
                expert_diversity = active_experts / expert_counts.shape[0]
                metrics[f"moe/expert_diversity"] = expert_diversity
                
                # Router collapse warning indicators
                if top_expert_usage > 0.5:  # More than 50% of tokens to one expert
                    metrics[f"moe/router_collapse_warning"] = 1.0
                else:
                    metrics[f"moe/router_collapse_warning"] = 0.0
                
                if expert_diversity < 0.5:  # Less than half of experts are used
                    metrics[f"moe/low_expert_diversity_warning"] = 1.0
                else:
                    metrics[f"moe/low_expert_diversity_warning"] = 0.0
        
        # Per-expert detailed metrics
        if "tokens_per_expert" in moe_metrics:
            tokens_per_expert = moe_metrics["tokens_per_expert"]
            for i, token_count in enumerate(tokens_per_expert):
                metrics[f"moe/expert_{i}_token_ratio"] = token_count.item()
        
        if "importance_per_expert" in moe_metrics:
            importance_per_expert = moe_metrics["importance_per_expert"]
            for i, importance in enumerate(importance_per_expert):
                metrics[f"moe/expert_{i}_importance"] = importance.item()
        
        # Load balancing quality metrics
        if "load_balance_loss" in moe_metrics:
            # Normalize by number of experts for better interpretation
            n_experts = getattr(model, 'config', None)
            if n_experts and hasattr(n_experts, 'num_experts'):
                normalized_lb_loss = moe_metrics["load_balance_loss"] / n_experts.num_experts
                metrics[f"moe/normalized_load_balance_loss"] = normalized_lb_loss.item()
        
        # Training stability metrics
        if step > 0 and "moe/load_balance_loss" in metrics:
            # Track load balance loss trend
            if "moe/prev_lb_loss" in metrics:
                lb_loss_change = metrics["moe/load_balance_loss"] - metrics["moe/prev_lb_loss"]
                metrics[f"moe/lb_loss_change"] = lb_loss_change
            
            metrics["moe/prev_lb_loss"] = metrics["moe/load_balance_loss"]
        
    except Exception as e:
        print(f"Warning: Failed to track MoE metrics: {e}")


def log_moe_metrics_to_wandb(metrics: dict, step: int, wandb_logger=None) -> None:
    """
    Log MoE metrics to WandB with proper organization and visualization.
    
    Args:
        metrics: Dictionary containing MoE metrics
        step: Current training step
        wandb_logger: WandB logger instance
    """
    if wandb_logger is None:
        return
    
    try:
        # Group MoE metrics by category for better organization
        moe_metrics = {}
        
        # Extract all MoE-related metrics
        for key, value in metrics.items():
            if key.startswith("moe/"):
                moe_metrics[key] = value
        
        if not moe_metrics:
            return
        
        # Log basic metrics
        basic_metrics = {
            "moe/load_balance_loss": moe_metrics.get("moe/load_balance_loss"),
            "moe/routing_entropy": moe_metrics.get("moe/routing_entropy"),
            "moe/expert_utilization": moe_metrics.get("moe/expert_utilization"),
            "moe/expert_diversity": moe_metrics.get("moe/expert_diversity"),
        }
        
        # Log router collapse indicators
        collapse_metrics = {
            "moe/gini_coefficient": moe_metrics.get("moe/gini_coefficient"),
            "moe/top_expert_usage_pct": moe_metrics.get("moe/top_expert_usage_pct"),
            "moe/bottom_expert_usage_pct": moe_metrics.get("moe/bottom_expert_usage_pct"),
            "moe/router_collapse_warning": moe_metrics.get("moe/router_collapse_warning"),
            "moe/low_expert_diversity_warning": moe_metrics.get("moe/low_expert_diversity_warning"),
        }
        
        # Log attention analysis
        attention_metrics = {
            "moe/max_expert_attention": moe_metrics.get("moe/max_expert_attention"),
            "moe/min_expert_attention": moe_metrics.get("moe/min_expert_attention"),
            "moe/attention_spread": moe_metrics.get("moe/attention_spread"),
            "moe/attention_variance": moe_metrics.get("moe/attention_variance"),
        }
        
        # Log per-expert metrics as histograms
        expert_metrics = {}
        for key, value in moe_metrics.items():
            if key.startswith("moe/expert_") and "_token_ratio" in key:
                expert_metrics[key] = value
            elif key.startswith("moe/expert_") and "_importance" in key:
                expert_metrics[key] = value
        
        # Log all metrics
        all_metrics = {**basic_metrics, **collapse_metrics, **attention_metrics}
        all_metrics = {k: v for k, v in all_metrics.items() if v is not None}
        
        if all_metrics:
            wandb_logger.log(all_metrics, step=step)
        
        # Log expert utilization histogram
        if expert_metrics:
            # Create histogram data for expert metrics
            expert_data = {}
            for key, value in expert_metrics.items():
                if value is not None:
                    expert_data[key] = value
            
            if expert_data:
                wandb_logger.log({"moe/expert_metrics_histogram": wandb_logger.Histogram(
                    list(expert_data.values()),
                    num_bins=10,
                    title="Expert Metrics Distribution"
                )}, step=step)
        
        # Log custom charts for MoE analysis
        if step % 100 == 0:  # Log charts less frequently
            # Create expert utilization heatmap
            if "moe/expert_0_token_ratio" in moe_metrics:
                expert_ratios = []
                expert_importances = []
                
                for i in range(8):  # Assuming 8 experts
                    token_ratio = moe_metrics.get(f"moe/expert_{i}_token_ratio", 0)
                    importance = moe_metrics.get(f"moe/expert_{i}_importance", 0)
                    expert_ratios.append(token_ratio)
                    expert_importances.append(importance)
                
                # Create custom chart
                wandb_logger.log({
                    "moe/expert_utilization_chart": wandb_logger.plot.line(
                        y=expert_ratios,
                        x=list(range(len(expert_ratios))),
                        title="Expert Token Utilization",
                        labels={"x": "Expert Index", "y": "Token Ratio"}
                    )
                }, step=step)
                
                wandb_logger.log({
                    "moe/expert_importance_chart": wandb_logger.plot.line(
                        y=expert_importances,
                        x=list(range(len(expert_importances))),
                        title="Expert Importance",
                        labels={"x": "Expert Index", "y": "Importance"}
                    )
                }, step=step)
    
    except Exception as e:
        print(f"Warning: Failed to log MoE metrics to WandB: {e}")


def _get_num_params(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    # unique_params = {p.data_ptr(): p for p in param_list}.values()
    return sum(p.numel() for p in param_list)


def _estimate_num_flop_per_token(num_params: int, model_config) -> int:
    l, h, q, t = (
        model_config.n_layer,
        getattr(model_config, "n_heads", model_config.num_attention_heads),
        model_config.head_size,
        model_config.block_size,
    )
    flop_per_token = 6 * num_params + 12 * l * h * q * t
    return flop_per_token * 3  # 1 fwd + 2 bwd


def _actually_measure_flops(model_config, objective, gradient_checkpointing, micro_batch_size=1) -> int:
    """Measure FLOP usage for a single sequence."""
    try:
        with torch.device("meta"):
            config_copy = copy.deepcopy(model_config)
            # # Annoying special rules for improper triton implementations
            config_copy.simple_ops = True
            config_copy.use_fused_head = False
            # construct a new model made up only of meta tensors:
            meta_model = config_copy.construct_model(objective=objective, gradient_checkpointing=gradient_checkpointing)
            x = torch.randint(0, config_copy.padded_vocab_size, (micro_batch_size, model_config.block_size))

            flop_counter = FlopCounterMode(display=not torch.distributed.is_initialized())
            with flop_counter:
                meta_model(input_ids=x, labels=x)["loss"].backward()
            measured_flops = flop_counter.get_total_flops()
            del meta_model, x
    except (NotImplementedError, AssertionError, RuntimeError) as e:
        print(
            "Cannot trace model with meta tensors for flop calculation, falling back on estimated flop count. "
            f"This may be (very) inaccurate for exotic archs. Original error: {e}"
        )
        with torch.device("meta"):
            meta_model = model_config.construct_model(
                objective=objective, gradient_checkpointing=gradient_checkpointing
            )
        num_params = sum(p.numel() for p in meta_model.parameters())
        measured_flops = 3 * _estimate_num_flop_per_token(num_params, model_config)
        print(f"FLOP estimate for this model is {measured_flops:,}, based on 3 * (6P + 12 LHS)")
    return measured_flops


def _get_peak_flops(fabric_precision, device_name: str) -> int:
    """Assuming tensor core usage for all nvidia cards"""
    if "32" in fabric_precision:
        multiplier = 0.5
    elif "64" in fabric_precision:
        multiplier = 0.25
    elif "8" in fabric_precision:
        multiplier = 2
    else:
        multiplier = 1

    if "MI250" in device_name:
        # https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
        flops = 383e12 / 2 * multiplier  # only one die counted

    elif "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        flops = 312e12 * multiplier
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            flops = 1979e12 * multiplier
        elif "PCIe" in device_name:
            flops = 756e12 * multiplier
        else:  # for SXM and other variants
            flops = 989e12 * multiplier
    elif "V100" in device_name:
        flops = 125e12 * multiplier  # sxm
    elif "RTX 6000 Ada" in device_name:
        flops = 210.6e12 * multiplier  # or 364.25 = 1457.0 / 2 / 2? thanks nvidia
    elif "A4000" in device_name:
        # 4000 performance is actually less clear
        # from this whitepaper, and estimated to be 88.45 TFLOP/s, based on it containing 192 tensor cores,
        # compared to 336 for the A6000.
        flops = 88.45e12 * multiplier
    elif "A5000" in device_name:
        flops = 111.1e12 * multiplier
    elif "A6000" in device_name:
        flops = 154.8e12 * multiplier
    elif "2080 Ti" in device_name:
        flops = 53.8e12 * multiplier
    elif "RTX 3050 Ti Laptop" in device_name:
        flops = 21.2e12  # 5.299 / 9.098 * 36.4 # :)
    else:  # for other GPU types, raise
        raise ValueError(f"Could not retrieve flops for device {device_name}.")
    return int(flops)  # ok up to 1e18


# Cache dictionary
model_cache = {}


def get_MFU_metrics(tokens_per_second, fabric, model, precision, measure_flops=True):
    if id(model) not in model_cache:
        model_param_count = _get_num_params(model)
        if measure_flops:
            measured_flops = _actually_measure_flops(model.config, model.objective, model.gradient_checkpointing)
            num_flop_per_token = measured_flops / model.config.block_size
        else:
            num_flop_per_token = _estimate_num_flop_per_token(model_param_count, model.config)

        peak_flops = _get_peak_flops(precision, torch.cuda.get_device_name(device=fabric.device))
        model_cache[id(model)] = {
            "num_params": model_param_count,
            "num_flop_per_token": num_flop_per_token,
            "peak_flops": peak_flops,
        }
    cache = model_cache[id(model)]
    flops = cache["num_flop_per_token"] * tokens_per_second
    mfu = flops / fabric.world_size / cache["peak_flops"]
    return cache["num_flop_per_token"], flops / 1e12, mfu


def standalone_measure_peak_flops(A100_PEAK_TFLOPS_FP16=312, dtype=torch.float16):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    bs = 16

    for max_size in [2048, 4096, 8192, 16384]:
        # Determine the largest matrix size that fits in memory
        # while True:
        #     try:
        #         torch.cuda.empty_cache()
        #         a = torch.randn(max_size, max_size, dtype=torch.float16, device="cuda")
        #         b = torch.randn(max_size, max_size, dtype=torch.float16, device="cuda")
        #         del a, b
        #         max_size *= 2
        #     except RuntimeError:
        #         max_size //= 2
        #         break

        # Create matrices
        a = torch.randn(bs, max_size, max_size, dtype=dtype, device="cuda")
        b = torch.randn(bs, max_size, max_size, dtype=dtype, device="cuda")

        # Warm-up run
        torch.bmm(a, b)
        torch.cuda.synchronize()

        # Measure time for matrix multiplication
        print("Starting new measurement cycle")
        num_runs = 100
        cooldown_time = 0.1

        times = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()  # type: ignore
            torch.bmm(a, b)
            end.record()  # type: ignore

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000)  # Convert to seconds

            # Cooldown step
            torch.cuda.empty_cache()  # Clear CUDA cache
            time.sleep(cooldown_time)  # Wait for cooldown_time seconds

        mean_time, std_time = torch.mean(torch.as_tensor(times)), torch.std(torch.as_tensor(times))

        # Calculate FLOPs
        flops = 2 * max_size**3 * bs / mean_time  # 2n^3 FLOPs for matrix multiplication
        tflops = flops / 1e12

        # Calculate MFU
        mfu = (tflops / A100_PEAK_TFLOPS_FP16) * 100

        print(f"Matrix size: {bs}x{max_size}x{max_size}")
        print(f"Time per multiplication: {mean_time:.6f} seconds, std={std_time:.6f}")
        print(f"Measured peak performance: {tflops:.2f} TFLOPS ({dtype})")
        print(f"A100 theoretical peak in float16: {A100_PEAK_TFLOPS_FP16:.2f} TFLOPS ({dtype})")
        print(f"Model FLOPs Utilization (MFU): {mfu:.2f}%")


if __name__ == "__main__":
    standalone_measure_peak_flops()

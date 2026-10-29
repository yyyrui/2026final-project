"""Analytic allocations in the unmodified TACTiS2 decoder, float32 only."""
import torch
import psutil


def estimate(series, config):
    history, horizon = config['context_length'], config['horizon']
    decoder = config['model']['copula_decoder']['attentional_copula']
    heads, layers, dimension = (decoder[k] for k in ('attention_heads','attention_layers','attention_dim'))
    predicted, observed = series*horizon, series*history
    # decoder.py::loss explicitly allocates [batch, heads, pred_tokens, all_tokens].
    one = heads*predicted*(predicted+observed)*4  # validation batch = 1
    bag = min(series, config['training']['bagging_size'])
    train_one = config['training']['batch_size']*heads*(bag*horizon)*(bag*(history+horizon))*4
    # decoder.py::sample keys_samples AND values_samples, allocated for each layer.
    sample_cache = 2*layers*config['sampling']['samples_per_call']*heads*predicted*dimension*4
    return dict(series=series, history_days=history, horizon_days=horizon,
                validation_one_dense_tensor_bytes=one,
                validation_one_dense_tensor_TiB=one/2**40,
                training_bag_one_dense_tensor_bytes=train_one,
                sampling_predicted_kv_cache_bytes=sample_cache,
                joint_autoregressive_steps_per_sample_call=predicted,
                validation_note='One mandatory tensor only, NOT total or peak memory. Several equally sized tensors coexist.',
                inference_note='Sample batching preserves all series per joint draw; no series partitioning. Encoder and temporary allocations excluded.')


def hardware(device):
    if device == 'cuda' and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return dict(device='cuda', name=torch.cuda.get_device_name(), free_bytes=free,total_bytes=total)
    memory = psutil.virtual_memory()
    return dict(device='cpu', cuda_available=torch.cuda.is_available(), free_bytes=memory.available,total_bytes=memory.total)


def blockers(estimate, capacity):
    reasons=[]
    # Conservative refusal, not a guarantee that every accepted configuration fits.
    if estimate['validation_one_dense_tensor_bytes']*4 > capacity['free_bytes']*.8:
        reasons.append('Unbagged validation dense attention exceeds device memory allowance (4x single tensor guard).')
    if estimate['sampling_predicted_kv_cache_bytes']*2 > capacity['free_bytes']*.8:
        reasons.append('Joint sample cache exceeds device memory allowance (2x cache guard).')
    return reasons

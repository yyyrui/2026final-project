"""Adapters for the unchanged local model source trees."""
from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def train_tactis(values, args, output):
    sys.path.insert(0, str(ROOT / 'TACTiS2' / 'TACTiS'))
    from tactis.model.tactis import TACTiS

    encoder = dict(attention_layers=1, attention_heads=1, attention_dim=16,
                   attention_feedforward_dim=16, dropout=0.0)
    config = dict(num_series=len(values), flow_series_embedding_dim=4,
                  copula_series_embedding_dim=4, flow_input_encoder_layers=1,
                  copula_input_encoder_layers=1, data_normalization='standardization',
                  loss_normalization='both', positional_encoding={'dropout': 0.0},
                  flow_temporal_encoder=encoder.copy(), copula_temporal_encoder=encoder.copy(),
                  copula_decoder=dict(min_u=0.05, max_u=0.95,
                      attentional_copula=dict(attention_heads=1, attention_layers=1,
                          attention_dim=8, mlp_layers=1, mlp_dim=16, resolution=10,
                          dropout=0.0, attention_mlp_class='_simple_linear_projection'),
                      dsf_marginal=dict(mlp_layers=1, mlp_dim=16, flow_layers=2, flow_hid_dim=16)),
                  skip_copula=True)
    model = TACTiS(**deepcopy(config)).to(args.device)
    data = torch.as_tensor(values, device=args.device)
    context, horizon = args.context_length, args.horizon
    cutoff = data.shape[1] - horizon  # last 7 training days are validation only
    if cutoff < context + horizon:
        raise ValueError('Not enough training days for context, forecast and validation')
    hist_time = torch.arange(context, device=args.device)[None, None, :]
    pred_time = torch.arange(context, context + horizon, device=args.device)[None, None, :]
    rng = np.random.default_rng(args.seed)
    log = []

    def loss_at(end):
        marginal, copula = model.loss(hist_time, data[None, :, end-context:end],
                                     pred_time, data[None, :, end:end+horizon])
        return (copula - marginal).mean()

    for stage in (1, 2):
        if stage == 2:
            model.initialize_stage2()
            model.to(args.device)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_('copula' in name)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        best_loss, best_state = float('inf'), None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for _ in range(args.steps_per_epoch):
                end = int(rng.integers(context, cutoff - horizon + 1))
                optimizer.zero_grad(set_to_none=True)
                loss = loss_at(end)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite TACTiS training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
            model.eval()
            # Fixed RNG for comparable stochastic copula validation across epochs.
            with torch.random.fork_rng(devices=([torch.cuda.current_device()] if args.device == 'cuda' else [])):
                torch.manual_seed(args.seed + 100)
                with torch.no_grad():
                    val = loss_at(cutoff).item()
            if not np.isfinite(val):
                raise RuntimeError('Nonfinite TACTiS validation loss')
            if val < best_loss:
                best_loss, best_state = val, deepcopy(model.state_dict())
            record = dict(stage=stage, epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=val)
            log.append(record)
            print(record, flush=True)
        model.load_state_dict(best_state)
    torch.save(dict(config=config, stage=2, model=model.state_dict(), training_log=log), output / 'tactis.pt')
    model.eval()
    with torch.no_grad():
        samples = model.sample(args.num_samples, hist_time, data[None, :, -context:], pred_time)
    return samples[0, :, -horizon:, :].cpu().numpy(), log


def load_chronos(args):
    sys.path.insert(0, str(ROOT / 'chronos' / 'chronos-forecasting' / 'src'))
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(args.chronos_model, device_map=args.device,
                                               torch_dtype=torch.float32, revision=getattr(args, "chronos_revision", None), cache_dir=str(ROOT / '.cache' / 'huggingface'))
    return pipeline


def predict_chronos(values, args, pipeline=None):
    if pipeline is None:
        pipeline = load_chronos(args)
    batches = []
    for start in range(0, len(values), args.chronos_batch_size):
        context = torch.from_numpy(values[start:start + args.chronos_batch_size, -args.context_length:].copy())
        with torch.no_grad():
            samples = pipeline.predict(context, prediction_length=args.horizon, num_samples=args.num_samples)
        batches.append(samples.cpu().numpy().transpose(0, 2, 1))
    return np.concatenate(batches)

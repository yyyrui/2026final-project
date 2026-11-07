"""Chronos-2 quantile inference on independent univariate sales series."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPEC = json.loads((ROOT / "chronos/model.json").read_text())
DEFAULT_MODEL = MODEL_SPEC["model_id"]
DEFAULT_REVISION = MODEL_SPEC["model_revision"]


def loading_options(args):
    return dict(revision=args.chronos2_revision,
                cache_dir=str(ROOT / ".cache/huggingface"),
                local_files_only=args.offline)


def validate_model(args):
    """Reject other Chronos families before starting expensive TACTiS2 training."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.chronos2_model, **loading_options(args))
    if (getattr(config, "chronos_pipeline_class", None) != "Chronos2Pipeline"
            or getattr(config, "architectures", None) != ["Chronos2Model"]):
        raise ValueError("Expected a Chronos-2 checkpoint (Chronos2Model / Chronos2Pipeline)")
    if args.context > config.chronos_config["context_length"]:
        raise ValueError("Requested context exceeds this Chronos-2 checkpoint's context length")
    return config


def predict_chronos2(panel, args):
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download

    config = validate_model(args)
    checkpoint = args.chronos2_model
    if not Path(checkpoint).is_dir():
        options = loading_options(args)
        # Resolve weights to the same commit as the validated config. Passing a
        # local snapshot also prevents the upstream LoRA probe from ignoring
        # local_files_only/revision and querying the remote main branch.
        options["revision"] = getattr(config, "_commit_hash", None) or args.chronos2_revision
        checkpoint = snapshot_download(
            repo_id=args.chronos2_model,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
            **options)
    pipeline = Chronos2Pipeline.from_pretrained(
        checkpoint, device_map=args.device, torch_dtype=torch.float32, local_files_only=True)
    if not isinstance(pipeline, Chronos2Pipeline):
        raise TypeError("Only Chronos2Pipeline is supported")
    levels = np.asarray(pipeline.quantiles, dtype=np.float64)
    median_indices = np.flatnonzero(levels == 0.5)
    if len(median_indices) != 1:
        raise ValueError("Chronos-2 must provide exactly one 0.5 quantile")
    median_index = int(median_indices[0])
    batches = []
    for start in range(0, len(panel.keys), args.chronos2_batch_size):
        stop = min(start + args.chronos2_batch_size, len(panel.keys))
        # [tasks, 1 variate, context]: do not reinterpret a batch as one
        # multivariate task or allow predictions to depend on batch neighbours.
        history = torch.from_numpy(panel.train[start:stop, -args.context:].copy()).unsqueeze(1)
        with torch.inference_mode():
            forecasts = pipeline.predict(
                history, prediction_length=args.horizon, batch_size=args.chronos2_batch_size,
                context_length=args.context, cross_learning=False)
        if len(forecasts) != stop - start:
            raise ValueError("Chronos-2 returned an incomplete batch")
        for forecast in forecasts:
            if tuple(forecast.shape) != (1, len(levels), args.horizon):
                raise ValueError(f"Unexpected Chronos-2 quantile shape: {forecast.shape}")
        batches.append(torch.cat(forecasts, dim=0).transpose(1, 2).float().cpu().numpy())
        print(f"Chronos-2: {stop}/{len(panel.keys)}", flush=True)
    quantiles = np.concatenate(batches, axis=0)
    if quantiles.shape != (len(panel.keys), args.horizon, len(levels)) or not np.isfinite(quantiles).all():
        raise ValueError("Invalid Chronos-2 quantile forecasts")
    # Quantile forecasts are not Monte Carlo samples. Select q=0.5 directly.
    np.savez_compressed(args.work_dir / "chronos2_quantiles.npz",
                        quantiles=quantiles, quantile_levels=levels)
    provenance = dict(
        family="Chronos-2", model_id=args.chronos2_model,
        requested_revision=args.chronos2_revision,
        resolved_revision=getattr(config, "_commit_hash", None),
        pipeline_class=type(pipeline).__name__,
        parameter_count=sum(p.numel() for p in pipeline.model.parameters()),
        package_version=importlib.metadata.version("chronos-forecasting"),
        quantile_levels=levels.tolist(), point_quantile=0.5,
        context_length=args.context, cross_learning=False, dtype="float32")
    local_model = Path(args.chronos2_model)
    if local_model.is_dir():
        provenance["local_files_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in local_model.iterdir()
            if path.is_file() and (path.name == "config.json" or path.suffix == ".safetensors")
        }
    (args.work_dir / "chronos2_model.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return quantiles[:, :, median_index]

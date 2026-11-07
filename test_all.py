#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""資料集內的全部時間序列（不抽樣、不分小組）：TACTiS2、Chronos-2 Small、M4 Naive 比較。

執行：python test_all.py
GPU：python test_all.py --device cuda
僅檢查資料：python test_all.py --prepare-only

本檔包含完整的資料選取、模型訓練／推論、評分與輸出流程。
預設 seed=1102。三模型共用序列、最後 28 天歷史與未來 7 天評估資料。
TACTiS2 每次更新使用全部選取序列；bagging_size=None，不建立分群模型。
Chronos-2 Small 使用預訓練權重；M4 Naive 逐條預測最後觀測值。

完成後：
- test_all.csv：三個模型各一列，依 MAE 排名，另列 RMSE、WAPE 與耗時。
- outputs/test_all/predictions.csv：每條序列、每天的真值與三模型預測。
- outputs/test_all/：選取 ID、設定、權重、訓練紀錄及分位數。

仍需保留 dataset/、TACTiS2/、chronos/、sktime-m4/，並安裝 requirements.txt。
相對路徑以本檔所在資料夾為準；已有結果不覆寫。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import MethodType

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
PROJECT = ROOT
MODE = 'all'
NUM_SERIES = None
DEFAULT_SEED = 1102
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "TACTiS2" / "TACTiS"))
sys.path.insert(0, str(PROJECT / "sktime-m4" / "scripts"))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

KEYS = ["city_id", "store_id", "product_id"]
MODEL_SPEC = json.loads((ROOT / "chronos/model.json").read_text())
DEFAULT_MODEL = MODEL_SPEC["model_id"]
DEFAULT_REVISION = MODEL_SPEC["model_revision"]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments(argv=None):
    """本檔固定使用 資料集內的全部時間序列（不抽樣、不分小組），參數可調整裝置、訓練與輸出位置。"""
    run_name = SCRIPT_PATH.stem
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT / "dataset/train.parquet")
    parser.add_argument("--eval-path", type=Path, default=PROJECT / "dataset/eval.parquet")
    parser.add_argument("--output", type=Path, help="three-model comparison CSV; defaults to this script name with .csv")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs" / run_name)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--context", type=positive, default=28)
    parser.add_argument("--horizon", type=positive, default=7)
    parser.add_argument("--epochs", type=positive, default=30, help="maximum epochs PER stage")
    parser.add_argument("--patience", type=positive, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--samples", type=positive, default=20, help="TACTiS2 samples only; Chronos-2 predicts quantiles directly")
    parser.add_argument("--samples-per-call", type=positive, default=1,
                        help="independent draws per call; every draw includes ALL selected series")
    parser.add_argument("--attention-block", type=positive, default=128,
                        help="query rows per attention calculation; does NOT group series")
    parser.add_argument("--chronos2-batch-size", type=positive, default=32,
                        help="inference batches for independent pretrained Chronos-2 series")
    parser.add_argument("--chronos2-model", default=DEFAULT_MODEL, help="Chronos-2 Hub model ID or local checkpoint directory")
    parser.add_argument("--chronos2-revision", help="Hub revision; defaults to the pinned revision for Chronos-2 Small")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    local_chronos2 = ROOT / args.chronos2_model
    if local_chronos2.is_dir():
        args.chronos2_model = str(local_chronos2.resolve())
    if args.chronos2_revision is None and args.chronos2_model == DEFAULT_MODEL:
        args.chronos2_revision = DEFAULT_REVISION
    args.mode = MODE
    args.num_series = NUM_SERIES
    if args.output is None:
        args.output = SCRIPT_PATH.with_suffix(".csv")
    if args.context < 2 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("context must be >= 2; learning-rate must be finite and positive")
    if not 0 <= args.seed < 2**63 - 20001:
        parser.error("seed must be a nonnegative integer smaller than 2**63 - 20001")
    for name in ("train_path", "eval_path", "output", "work_dir"):
        path = getattr(args, name)
        setattr(args, name, (path if path.is_absolute() else ROOT / path).resolve())
    if args.output == args.work_dir / "predictions.csv":
        parser.error("--output is the comparison table; choose a path different from WORK_DIR/predictions.csv")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()



# %% 資料讀取、完整序列選取與檢查
@dataclass
class SalesPanel:
    keys: pd.DataFrame
    train_dates: pd.DatetimeIndex
    eval_dates: pd.DatetimeIndex
    train: np.ndarray  # [series, day]
    actual: np.ndarray
    total_series: int


def load_sales(train_path, eval_path, num_series=NUM_SERIES, seed=DEFAULT_SEED, horizon=7):
    """Scan Parquet in bounded batches; num_series=None keeps every series.

    Sampling uses sorted TRAIN IDs only, without inspecting evaluation targets.
    No individual observations are sampled and no missing dates are imputed.
    """
    import pyarrow.parquet as pq

    def all_keys(path):
        found = set()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=KEYS):
            frame = batch.to_pandas()
            if frame.isna().any().any():
                raise ValueError(f"{path}: null series IDs")
            found.update(frame.drop_duplicates().itertuples(index=False, name=None))
        if not found:
            raise ValueError(f"{path}: empty data")
        return pd.DataFrame(sorted(found), columns=KEYS)

    for path in (train_path, eval_path):
        with Path(path).open("rb") as stream:
            if stream.read(4) != b"PAR1":
                raise ValueError(f"{path}: not a Parquet file; run git lfs pull to download the dataset")
    if (num_series is not None and num_series < 1) or horizon < 1:
        raise ValueError("num_series and horizon must be positive")
    keys = all_keys(train_path)
    if num_series is not None and num_series > len(keys):
        raise ValueError(f"Requested {num_series} series, only {len(keys)} available")
    if num_series is None:
        selected = keys.copy()  # Full mode never samples or silently caps the panel.
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(keys), num_series, replace=False))
        selected = keys.iloc[indices].reset_index(drop=True)
    if not keys.equals(all_keys(eval_path)):
        raise ValueError("Train/eval series keys differ")

    def selected_rows(path):
        pieces = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=KEYS + ["dt", "sale_amount"]):
            part = batch.to_pandas().merge(selected, on=KEYS, how="inner", validate="many_to_one")
            if len(part):
                pieces.append(part)
        if not pieces:
            raise ValueError(f"{path}: no selected observations")
        frame = pd.concat(pieces, ignore_index=True)
        if frame.isna().any().any() or frame.duplicated(KEYS + ["dt"]).any():
            raise ValueError(f"{path}: null data or duplicate series/date observations")
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
        if not frame.dt.eq(frame.dt.dt.normalize()).all():
            raise ValueError(f"{path}: timestamps must be daily dates")
        # Check again after parsing in case different strings represented one day.
        if frame.duplicated(KEYS + ["dt"]).any():
            raise ValueError(f"{path}: duplicate parsed series/date observations")
        if not np.isfinite(frame.sale_amount.to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: nonfinite sales values")
        return frame

    train, evaluation = selected_rows(train_path), selected_rows(eval_path)
    train_dates = pd.DatetimeIndex(sorted(train.dt.unique()))
    eval_dates = pd.DatetimeIndex(sorted(evaluation.dt.unique()))
    if not train_dates.equals(pd.date_range(train_dates[0], train_dates[-1], freq="D")):
        raise ValueError("Training dates are not consecutive")
    if not eval_dates.equals(pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=horizon)):
        raise ValueError(f"Evaluation must cover the next {horizon} days after training")
    index = pd.MultiIndex.from_frame(selected)

    def matrix(frame, dates):
        if not frame.groupby(KEYS).size().reindex(index, fill_value=0).eq(len(dates)).all():
            raise ValueError("missing daily observations for selected series; no implicit filling")
        values = frame.pivot(index=KEYS, columns="dt", values="sale_amount").reindex(
            index=index, columns=dates).to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Panel contains missing/nonfinite values after float32 conversion")
        return values

    return SalesPanel(selected, train_dates, eval_dates, matrix(train, train_dates),
                      matrix(evaluation, eval_dates), len(keys))


# %% TACTiS2：單一聯合模型，沒有序列分群或 bagging
def recompute(function, *inputs):
    # Recompute a query block during backward instead of retaining its N x N
    # scores. All blocks contribute to ONE loss and ONE optimizer update.
    if torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
        return checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def bounded_self_attention(layer, x, attn_mask=None, key_padding_mask=None, is_causal=False):
    """Equivalent unmasked self-attention for the local temporal encoders.

    Every query still attends to every key. Only score-matrix allocation is
    bounded. Transformer residuals, norms and feed-forward layers are unchanged.
    """
    if attn_mask is not None or key_padding_mask is not None or is_causal:
        raise ValueError("This adapter supports the local unmasked temporal encoder only")
    attention = layer.self_attn
    if attention.batch_first or attention.bias_k is not None or attention.add_zero_attn:
        raise ValueError("Unexpected upstream MultiheadAttention configuration")
    length, batch, width = x.shape
    heads, dimension = attention.num_heads, width // attention.num_heads
    q, k, v = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)

    def arrange(tensor):
        return tensor.reshape(length, batch, heads, dimension).permute(1, 2, 0, 3).contiguous()

    q, k, v = map(arrange, (q, k, v))
    dropout = attention.dropout if layer.training else 0.0

    def attend(query, keys, values):
        return F.scaled_dot_product_attention(query, keys, values, dropout_p=dropout)

    parts = [recompute(attend, q[:, :, start:start + layer.query_block], k, v)
             for start in range(0, length, layer.query_block)]
    result = torch.cat(parts, dim=2).permute(2, 0, 1, 3).reshape(length, batch, width)
    return layer.dropout1(F.linear(result, attention.out_proj.weight, attention.out_proj.bias))


def bounded_copula_loss(copula, hist_encoded, hist_true_u, pred_encoded, pred_true_u,
                        num_series, num_timesteps):
    """Same ordering, strict causal mask and density as local AttentionalCopula.loss.

    Checkpoint whole query blocks, including every decoder layer. Keys and values
    always contain ALL series; gradients from every query block accumulate into
    the same encoders and projections before the single optimizer.step().
    """
    batch, variables, _ = pred_encoded.shape
    history = hist_encoded.shape[1]
    if variables != num_series * num_timesteps:
        raise ValueError("Copula dimensions disagree")
    inputs = torch.cat((torch.cat((hist_encoded, hist_true_u.unsqueeze(-1)), dim=-1),
                        torch.cat((pred_encoded, pred_true_u.unsqueeze(-1)), dim=-1)), dim=1)
    projections = []
    for layer in range(copula.attention_layers):
        projections.extend((torch.stack([mlp(inputs) for mlp in copula.key_creators[layer]], dim=1),
                            torch.stack([mlp(inputs) for mlp in copula.value_creators[layer]], dim=1)))
    total = pred_encoded.sum(dim=(1, 2)) * 0.0
    # The upstream density treats variable zero as Uniform(0,1), hence zero loss.
    for start in range(1, variables, copula.query_block):
        stop = min(start + copula.query_block, variables)

        def block_loss(encoded, truth, *keys_values, start=start, stop=stop):
            value = copula.dimension_shifting_layer(encoded)
            query_positions = torch.arange(start, stop, device=encoded.device)
            key_positions = torch.arange(history + variables, device=encoded.device)
            # History is always visible; predicted variables strictly BEFORE q.
            allowed = key_positions[None, :] < (history + query_positions[:, None])
            for layer in range(copula.attention_layers):
                query = value.reshape(batch, stop - start, copula.attention_heads,
                                      copula.attention_dim).transpose(1, 2).contiguous()
                attended = F.scaled_dot_product_attention(
                    query, keys_values[2 * layer], keys_values[2 * layer + 1],
                    attn_mask=allowed, dropout_p=0.0)
                attended = attended.transpose(1, 2).reshape(batch, stop - start, -1)
                value = copula.attention_layer_norms[layer](
                    value + copula.attention_dropouts[layer](attended))
                value = copula.feed_forward_layer_norms[layer](value + copula.feed_forwards[layer](value))
            target = (truth * copula.resolution).floor().long().clamp(0, copula.resolution - 1)
            logprob = F.log_softmax(copula.dist_extractors(value), dim=-1) + math.log(copula.resolution)
            return -logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        total = total + recompute(block_loss, pred_encoded[:, start:stop],
                                  pred_true_u[:, start:stop], *projections)
    return total


def install_bounded_attention(model, block):
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            module.query_block = block
            module._sa_block = MethodType(bounded_self_attention, module)
    if not model.skip_copula:
        model.decoder.copula.query_block = block
        model.decoder.copula.loss = MethodType(bounded_copula_loss, model.decoder.copula)


def model_config(n):
    # A compact TACTiS-2 architecture, not the paper's traffic benchmark config.
    encoder = dict(attention_layers=1, attention_heads=2, attention_dim=8,
                   attention_feedforward_dim=32, dropout=0.0)
    return dict(num_series=n, bagging_size=None,
                flow_series_embedding_dim=8, copula_series_embedding_dim=8,
                flow_input_encoder_layers=2, copula_input_encoder_layers=2,
                data_normalization="standardization", loss_normalization="both",
                positional_encoding={"dropout": 0.0},
                flow_temporal_encoder=encoder.copy(), copula_temporal_encoder=encoder.copy(),
                copula_decoder=dict(min_u=0.05, max_u=0.95,
                    attentional_copula=dict(attention_heads=1, attention_layers=1, attention_dim=16,
                        mlp_layers=2, mlp_dim=32, resolution=20, dropout=0.0,
                        attention_mlp_class="_simple_linear_projection"),
                    dsf_marginal=dict(mlp_layers=2, mlp_dim=32, flow_layers=2, flow_hid_dim=16)),
                skip_copula=True, experiment_mode="forecasting")


def train_tactis(panel, args):
    from tactis.model.tactis import TACTiS

    seed_all(args.seed)
    config = model_config(len(panel.keys))
    model = TACTiS(**deepcopy(config)).to(args.device)
    install_bounded_attention(model, args.attention_block)
    values = torch.from_numpy(panel.train).to(args.device)
    context, horizon = args.context, args.horizon
    cutoff = values.shape[1] - horizon  # final training week for validation only
    ends = np.arange(context, cutoff - horizon + 1)
    if not len(ends):
        raise ValueError("Not enough training history for context, training targets and validation")
    ht = torch.arange(context, device=args.device)[None, None, :]
    pt = torch.arange(context, context + horizon, device=args.device)[None, None, :]
    rng = np.random.default_rng(args.seed)
    logs = []

    def loss_at(end, stage):
        marginal, copula = model.loss(ht, values[None, :, end-context:end],
                                      pt, values[None, :, end:end+horizon])
        return (-marginal if stage == 1 else copula).mean()

    for stage in (1, 2):
        if stage == 2:
            model.initialize_stage2()
            model.to(args.device)
            install_bounded_attention(model, args.attention_block)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_("copula" in name)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=args.learning_rate)
        best, stale = float("inf"), 0
        path = args.work_dir / f"best_stage{stage}.pt"
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            losses = []
            # Each epoch visits EVERY eligible daily training window, in random
            # order. Every optimizer step jointly contains ALL selected series.
            for step, end in enumerate(rng.permutation(ends), start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_at(int(end), stage)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite TACTiS-2 training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
                if step == 1 or step % 10 == 0 or step == len(ends):
                    print(f"TACTiS-2 stage {stage}, epoch {epoch}, window {step}/{len(ends)}, "
                          f"all {len(panel.keys)} series, loss={loss.item():.6f}", flush=True)
            model.eval()
            with torch.no_grad():
                validation = loss_at(cutoff, stage).item()
            if not math.isfinite(validation):
                raise RuntimeError("Nonfinite TACTiS-2 validation loss")
            if validation < best:
                best, stale = validation, 0
                torch.save(model.state_dict(), path)
            else:
                stale += 1
            logs.append(dict(stage=stage, epoch=epoch, train_loss=float(np.mean(losses)),
                             validation_loss=validation, seconds=time.perf_counter() - started,
                             joint_series=len(panel.keys), windows=len(ends)))
            write_json(args.work_dir / "training_log.json", logs)
            print(logs[-1], flush=True)
            if stale >= args.patience:
                break
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
    torch.save(dict(config=config, stage=2, model=model.state_dict(),
                    selected_keys=panel.keys.to_dict("records"), seed=args.seed,
                    attention_block=args.attention_block), args.work_dir / "tactis2.pt")
    model.eval()
    draws = []
    for start in range(0, args.samples, args.samples_per_call):
        seed_all(args.seed + 10000 + start)
        count = min(args.samples_per_call, args.samples - start)
        print(f"TACTiS-2 joint forecast: samples {start+1}-{start+count}/{args.samples}; "
              f"{len(panel.keys) * horizon:,} autoregressive variables per draw", flush=True)
        with torch.no_grad():
            result = model.sample(count, ht, values[None, :, -context:], pt)
        draws.append(result[0, :, -horizon:, :].cpu().numpy())
    samples = np.concatenate(draws, axis=-1)
    validate_samples(samples, len(panel.keys), args)
    np.savez_compressed(args.work_dir / "tactis2_samples.npz", samples=samples)
    return np.median(samples, axis=-1)


def validate_samples(samples, n, args):
    if samples.shape != (n, args.horizon, args.samples) or not np.isfinite(samples).all():
        raise ValueError(f"Invalid forecast samples: {samples.shape}")


# %% Chronos-2 Small：固定模型版本，直接預測分位數
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


# %% 三模型的共同評分與 CSV 輸出
def score(actual, prediction):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    denominator = np.abs(actual.astype(np.float64)).sum()
    return dict(MAE=float(np.abs(error).mean()), RMSE=float(np.sqrt(np.square(error).mean())),
                WAPE=float(np.abs(error).sum() / denominator) if denominator else None,
                observations=int(error.size))


def prediction_frame(panel, predictions):
    n, horizon = panel.actual.shape
    frame = panel.keys.loc[panel.keys.index.repeat(horizon)].reset_index(drop=True)
    frame["dt"] = np.tile(panel.eval_dates.to_numpy(), n)
    frame["actual"] = panel.actual.reshape(-1)
    for name, prediction in predictions.items():
        if prediction.shape != (n, horizon) or not np.isfinite(prediction).all():
            raise ValueError(f"{name}: incomplete/nonfinite predictions")
        frame[name] = prediction.reshape(-1)
    return frame


def write_csv(path, frame):
    """以 UTF-8 BOM 輸出，可用 Excel 開啟；完成寫入後才換成正式檔名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(path)


# %% 執行流程：準備資料 → 三模型 → 比較成績 → 同名 CSV

def main(argv=None):
    args = arguments(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch / NVIDIA GPU unavailable. Install CUDA PyTorch or use --device cpu.")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.work_dir.exists() or (args.output.exists() and not args.prepare_only):
        raise FileExistsError("Existing results are protected. Choose a new --work-dir and --output.")
    # Check dependency availability before an expensive run.
    if not args.prepare_only:
        print("[Startup] Checking Chronos-2 dependencies and checkpoint family...", flush=True)
        from chronos import Chronos2Pipeline  # noqa: F401
        validate_model(args)
        print("[Startup] Checking sktime dependencies...", flush=True)
        from sktime.forecasting.naive import NaiveForecaster  # noqa: F401
        print("[Startup] Model dependencies loaded successfully.", flush=True)
    print("[Data] Scanning Parquet files and selecting complete series...", flush=True)
    panel = load_sales(args.train_path, args.eval_path, num_series=args.num_series,
                              seed=args.seed, horizon=args.horizon)
    if panel.train.shape[1] < args.context + 2 * args.horizon:
        raise ValueError("History is too short for the requested context and validation holdout")
    if not args.prepare_only:
        print(f"One joint TACTiS-2 model: {len(panel.keys):,} series; no grouping or bagging. "
              "Exact attention remains quadratic; full training and sampling may take substantial time and memory.", flush=True)
    args.work_dir.mkdir(parents=True)
    panel.keys.to_csv(args.work_dir / "selected_series.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(args.work_dir / "selected_data.npz", train=panel.train, actual=panel.actual,
                        train_dates=panel.train_dates.to_numpy(), eval_dates=panel.eval_dates.to_numpy())
    from sales_naive import source_spec
    versions = {}
    for package in ("numpy", "pandas", "pyarrow", "torch", "einops", "sktime", "chronos-forecasting", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    total_series=panel.total_series, selected_series=len(panel.keys),
                    train_shape=list(panel.train.shape), eval_shape=list(panel.actual.shape),
                    data_sha256={str(p): file_hash(p) for p in (args.train_path, args.eval_path)},
                    selected_series_sha256=file_hash(args.work_dir / "selected_series.csv"),
                    source_sha256={str(p): file_hash(p) for p in
                                   (SCRIPT_PATH,
                                    ROOT / "requirements.txt", ROOT / "sktime-m4/scripts/sales_naive.py",
                                    ROOT / "chronos/model.json",
                                    ROOT / "chronos/chronos-forecasting/src/chronos/chronos2/pipeline.py",
                                    PROJECT / "TACTiS2/TACTiS/tactis/model/tactis.py",
                                    PROJECT / "TACTiS2/TACTiS/tactis/model/decoder.py",
                                    PROJECT / "TACTiS2/TACTiS/tactis/model/encoder.py")},
                    versions=versions, tactis_config=model_config(len(panel.keys)),
                    chronos2=dict(model_id=args.chronos2_model, revision=args.chronos2_revision,
                                  family="Chronos-2", point_quantile=0.5, cross_learning=False),
                    sktime_m4_baseline=source_spec(),
                    protocol=dict(training="one joint model, no series groups, no bagging; all eligible windows each epoch",
                        memory="exact attention query blocks with checkpoint recomputation; every query sees all allowed keys",
                        validation="last horizon days of train only; best checkpoint per stage; no eval tuning or refit",
                        forecast_history=f"same final {args.context} training days for all three models",
                        chronos2="pretrained Chronos-2; no fine-tuning; univariate inputs; cross_learning=False",
                        point_forecast="TACTiS2 sample median; Chronos-2 q=0.5; Naive deterministic; no clipping",
                        ranking="MAE ascending (primary); RMSE and WAPE also reported; lower is better"))
    write_json(args.work_dir / "manifest.json", manifest)
    write_json(args.work_dir / "status.json", {"status": "prepared"})
    print(f"Selected {len(panel.keys):,}/{panel.total_series:,} series, seed={args.seed}; "
          f"train={panel.train.shape}, test={panel.actual.shape}", flush=True)
    if args.prepare_only:
        print(f"Data preparation complete: {args.work_dir}. No model has been trained.", flush=True)
        return
    predictions, metrics = {}, []
    try:
        from sales_naive import forecast_sales
        jobs = (("tactis2", lambda: train_tactis(panel, args)),
                ("chronos2", lambda: predict_chronos2(panel, args)),
                ("sktime_m4_naive", lambda: forecast_sales(panel.train, context_length=args.context,
                    horizon=args.horizon, progress=lambda done, total: print(f"sktime Naive: {done}/{total}", flush=True))))
        for name, run in jobs:
            write_json(args.work_dir / "status.json", {"status": "running", "model": name})
            started = time.perf_counter()
            prediction = run()
            prediction_frame(panel, {name: prediction})  # validates complete shared coverage
            predictions[name] = prediction
            np.save(args.work_dir / f"{name}_predictions.npy", prediction)
            metrics.append(dict(model=name, num_series=len(panel.keys), seed=args.seed, horizon=args.horizon,
                                **score(panel.actual, prediction),
                                total_seconds=time.perf_counter() - started))
            pd.DataFrame(metrics).to_csv(args.work_dir / "metrics_partial.csv", index=False, encoding="utf-8-sig")
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        # 只有三個模型全部成功，才輸出正式比較表與完整逐筆預測。
        prediction_path = args.work_dir / "predictions.csv"
        frame = prediction_frame(panel, predictions)
        write_csv(prediction_path, frame)
        table = pd.DataFrame(metrics).sort_values("MAE", kind="stable").reset_index(drop=True)
        table.insert(0, "MAE_rank", np.arange(1, len(table) + 1))
        write_csv(args.work_dir / "metrics.csv", table)
        write_csv(args.output, table)
        write_json(args.work_dir / "status.json", {"status": "complete", "output": str(args.output),
                   "predictions": str(prediction_path), "prediction_rows": len(frame),
                   "best_model_by_MAE": str(table.iloc[0].model)})
        print(table.to_string(index=False), flush=True)
        print(f"Comparison: {args.output}\nPredictions: {prediction_path}", flush=True)
    except BaseException as error:
        write_json(args.work_dir / "status.json", {"status": "failed", "error": str(error),
                   "completed_models": list(predictions)})
        raise



if __name__ == "__main__":
    main()

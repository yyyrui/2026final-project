"""Project-relative paths for the sales experiment, separate from M4 settings."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "dataset" / "train.parquet"
EVAL_PATH = ROOT / "dataset" / "eval.parquet"
TACTIS_RESULTS_DIR = ROOT / "outputs" / "full-comparison"
OUTPUT_DIR = ROOT / "outputs" / "compare-10-30_SKTime"
REPORT_PATH = ROOT / "compare 10-30_SKTime.md"


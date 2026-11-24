"""Experiment configuration shared by all pipeline steps."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKSUMS = ROOT / "data" / "checksums.json"
WORK = ROOT / "work"
MODELS_DIR = WORK / "models"
GENERATED = ROOT / "generated"
BUILD = ROOT / "build"
FIRMWARE = ROOT / "firmware"
HOST_DIR = ROOT / "host"
RESULTS = ROOT / "results"

RANDOM_STATE = 42

DATASETS = ["occupancy", "smoke", "gasdrift", "har"]

# Gas Sensor Array Drift: batches used for training; the rest are test.
GASDRIFT_TRAIN_BATCHES = range(1, 8)  # batches 1-7 train, 8-10 test

# Smoke Detection: a new recording session starts when the UTC gap exceeds this.
SMOKE_SESSION_GAP_S = 60
SMOKE_TEST_SIZE = 0.3

MODELS = ["dt5", "dt10", "rf10", "rf50", "gnb", "svc", "xgb"]

TOOLS = ["emlearn", "micromlgen", "m2cgen", "m2cgen_float"]

# Which (model, tool) pairs are attempted. Unsupported pairs are skipped,
# supported pairs that fail are recorded as conversion errors.
SUPPORTED = {
    "emlearn": {"dt5", "dt10", "rf10", "rf50", "gnb"},
    "micromlgen": {"dt5", "dt10", "rf10", "rf50", "gnb", "svc", "xgb"},
    "m2cgen": {"dt5", "dt10", "rf10", "rf50", "svc", "xgb"},
    "m2cgen_float": {"dt5", "dt10", "rf10", "rf50", "svc", "xgb"},
}

# Firmware build profiles: name -> (IDF target, CPU MHz, extra sdkconfig defaults)
PROFILES = {
    "esp32_240": ("esp32", 240, []),
    "esp32_160": ("esp32", 160, ["sdkconfig.160mhz"]),
    "esp32c3_160": ("esp32c3", 160, []),
    "esp32s3_240": ("esp32s3", 240, []),
    "esp32s3_160": ("esp32s3", 160, ["sdkconfig.160mhz"]),
}

BASELINE = "baseline"


def config_id(dataset: str, model: str, tool: str) -> str:
    return f"{dataset}/{model}/{tool}"


def generated_dir(dataset: str, model: str, tool: str) -> Path:
    return GENERATED / dataset / model / tool


def iter_configs(datasets=None, models=None, tools=None, include_baseline=False):
    """Yield (dataset, model, tool) for every attempted configuration."""
    for ds in datasets or DATASETS:
        if include_baseline:
            yield ds, BASELINE, BASELINE
        for model in models or MODELS:
            for tool in tools or TOOLS:
                if model in SUPPORTED[tool]:
                    yield ds, model, tool

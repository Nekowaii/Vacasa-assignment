import datetime
import json
import logging
import shutil
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from utils import percentage_change

MODEL_DIR = Path("models")


def is_new_promoted_model(new_scores: dict[str, float], scoring: str, threshold_pct: int) -> bool:
    promoted_scores = get_promoted_scores()
    if not promoted_scores or scoring not in promoted_scores:
        return True

    return percentage_change(new_scores[scoring], promoted_scores[scoring]) > threshold_pct


def save_promoted_model(pipeline: Pipeline, path: Path = MODEL_DIR, **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    iso_datetime = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%fZ")

    joblib.dump(pipeline, path / f"model_{iso_datetime}.pkl")

    with open(path / f"model_{iso_datetime}.json", "w") as f:
        json.dump(kwargs, f)


def load_promoted_model(metadata_only: bool = False, path: Path = MODEL_DIR):
    metadata = sorted(Path(path).glob("model_*.json"))

    if metadata_only:
        if metadata:
            with open(metadata[-1], "r") as f:
                return None, json.load(f)
        else:
            return None, None

    models = sorted(Path(path).glob("model_*.pkl"))

    if models and metadata:
        with open(models[-1], "rb") as f1, open(metadata[-1], "r") as f2:
            return joblib.load(f1), json.load(f2)

    raise FileNotFoundError("Model wasn't trained yet. Please use train.py first.")


def get_promoted_scores() -> dict[str, float] | None:
    _, metadata = load_promoted_model(metadata_only=True)
    if metadata:
        return metadata.get("scores", {})


def models_comparison_report(new_scores, scoring):
    promoted_scores = get_promoted_scores()
    if not promoted_scores:
        return

    for metric in promoted_scores:
        pct_change = percentage_change(new_scores[metric], promoted_scores[metric])
        logging.info(
            f"Metric [{metric}{'*' if scoring == metric else ''}]: {new_scores[metric]:.4} ({pct_change:+.2f}%)"
        )


def clean_models():
    if MODEL_DIR.exists():
        logging.info("Cleaning promoted models...")
        shutil.rmtree(MODEL_DIR)

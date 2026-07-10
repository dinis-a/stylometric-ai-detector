"""AI vs Human text prediction using stylometric features and a trained Random Forest model.

The model is hosted on Hugging Face and downloaded/cached on first use.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, Optional

import joblib
from huggingface_hub import hf_hub_download

from stylometric_ai_detector.features import extract_stylometric_features

logger = logging.getLogger(__name__)

# Feature order matching the trained model
_FEATURE_ORDER = [
    "char_count",
    "word_count",
    "avg_word_len",
    "punct_count",
    "sentence_count",
    "avg_sentence_len",
    "upper_case_count",
    "title_case_count",
]

# Hugging Face model repo configuration
_HF_REPO_ID = os.environ.get("HF_MODEL_REPO", "dinisds/stylometric-ai-detector")
_HF_FILENAME = "random_forest_stylometric_model.joblib"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "stylometric-ai-detector")

_model = None


def _load_model() -> Any:
    """Lazy-load the model, downloading from Hugging Face on first use.

    Lookup order:
    1. Package data directory (bundled in wheel / local dev)
    2. User cache directory (~/.cache/stylometric-ai-detector/)
    3. Download from Hugging Face

    Returns:
        The loaded scikit-learn Random Forest model.
    """
    global _model
    if _model is None:
        os.makedirs(_CACHE_DIR, exist_ok=True)

        local_path = None

        # 1. Check package data directory (bundled model or local dev)
        pkg_path = os.path.join(os.path.dirname(__file__), "data", _HF_FILENAME)
        if os.path.exists(pkg_path):
            local_path = pkg_path

        # 2. Check user cache
        if local_path is None:
            cache_path = os.path.join(_CACHE_DIR, _HF_FILENAME)
            if os.path.exists(cache_path):
                local_path = cache_path

        # 3. Download from Hugging Face
        if local_path is None:
            logger.info("Downloading model from Hugging Face: %s", _HF_REPO_ID)
            local_path = hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=_HF_FILENAME,
                cache_dir=_CACHE_DIR,
            )

        _model = joblib.load(local_path)
    return _model


def _features_to_array(features_dict: Dict[str, float]) -> list[list[float]]:
    """Convert a feature dict to the ordered array expected by the model."""
    return [[features_dict[k] for k in _FEATURE_ORDER]]


def predict(
    text: Optional[str] = None,
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Predict whether a text was written by AI or a human.

    Provide either `text` (raw string) or `features` (pre-computed feature dict).
    If both are provided, `text` takes precedence and features are extracted from it.

    The model is downloaded from Hugging Face on first use and cached locally.

    Args:
        text: Raw text string to classify.
        features: Pre-computed stylometric feature dict (from extract_stylometric_features).

    Returns:
        dict with:
            - label: "AI" or "Human"
            - probability: Confidence score for the predicted class (float between 0 and 1).

    Raises:
        ValueError: If neither text nor features is provided.
        OSError: If the model cannot be downloaded from Hugging Face.
    """
    if text is not None:
        features = extract_stylometric_features(text)
    elif features is None:
        raise ValueError("Either 'text' or 'features' must be provided.")

    model = _load_model()
    X = _features_to_array(features)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

    label = "AI" if prediction == 1.0 else "Human"

    return {
        "label": label,
        "probability": float(max(probabilities)),
    }

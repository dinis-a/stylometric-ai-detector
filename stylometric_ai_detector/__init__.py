"""stylometric-ai-detector — AI vs Human text detection using stylometric features.

A baseline Random Forest classifier that distinguishes AI-generated text from
human-written text using 8 surface-level stylometric features (character counts,
word lengths, punctuation density, sentence structure, and capitalization).

Provides two public functions:
    - :func:`extract_stylometric_features` — Extract 8 stylometric features from any text.
    - :func:`predict` — Classify text as "AI" or "Human" with a confidence score.

The model is auto-downloaded from Hugging Face on first use.
"""

from stylometric_ai_detector.features import extract_stylometric_features
from stylometric_ai_detector.predict import predict

__all__ = ["extract_stylometric_features", "predict"]
__version__ = "0.2.4"

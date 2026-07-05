"""Tests for the predict function."""

import os

import pytest

# Skip predict tests if model is not cached locally (neither in package nor HF cache)
_PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "stylometric_ai_detector", "data")
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "stylometric-ai-detector")
_MODEL_FILENAME = "random_forest_stylometric_model.joblib"
_MODEL_AVAILABLE = os.path.exists(os.path.join(_PKG_DIR, _MODEL_FILENAME)) or os.path.exists(
    os.path.join(_CACHE_DIR, _MODEL_FILENAME)
)

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Model not cached locally — run predict() once to download from Hugging Face",
)


def test_predict_with_text():
    """Prediction should work with raw text input."""
    from stylometric_ai_detector import predict

    result = predict(text="This is a sample sentence for testing purposes.")
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ("AI", "Human")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_with_features():
    """Prediction should work with pre-computed feature dict."""
    from stylometric_ai_detector import extract_stylometric_features, predict

    features = extract_stylometric_features(
        "The quick brown fox jumps over the lazy dog. It was a sunny day."
    )
    result = predict(features=features)
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ("AI", "Human")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_text_takes_precedence():
    """When both text and features are given, text should be used."""
    from stylometric_ai_detector import predict

    result = predict(
        text="Hello world.",
        features={
            "char_count": 999,
            "word_count": 999,
            "avg_word_len": 99.0,
            "punct_count": 99,
            "sentence_count": 99,
            "avg_sentence_len": 99.0,
            "upper_case_count": 99,
            "title_case_count": 99,
        },
    )
    assert result["label"] in ("AI", "Human")


def test_predict_no_input_raises():
    """Calling predict with no arguments should raise ValueError."""
    from stylometric_ai_detector import predict

    with pytest.raises(ValueError, match="Either 'text' or 'features' must be provided"):
        predict()


def test_predict_consistency():
    """Same input should produce the same output."""
    from stylometric_ai_detector import predict

    text = "The theory of relativity explains the relationship between space and time."
    result1 = predict(text=text)
    result2 = predict(text=text)
    assert result1 == result2

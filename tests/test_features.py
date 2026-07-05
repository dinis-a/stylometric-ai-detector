"""Tests for stylometric feature extraction."""

from stylometric_ai_detector import extract_stylometric_features


def test_empty_string():
    """Empty or whitespace-only strings should return all zeros."""
    result = extract_stylometric_features("")
    assert result["char_count"] == 0
    assert result["word_count"] == 0
    assert result["avg_word_len"] == 0.0


def test_non_string_input():
    """Non-string input should return all zeros."""
    result = extract_stylometric_features(None)
    assert result["char_count"] == 0
    assert result["word_count"] == 0


def test_simple_sentence():
    """Basic feature extraction on a simple sentence."""
    text = "Hello world. This is a test."
    result = extract_stylometric_features(text)

    assert result["char_count"] == len(text)
    # "Hello", "world.", "This", "is", "a", "test."
    assert result["word_count"] == 6
    assert result["sentence_count"] == 2
    # Punctuation: '.' (after world) and '.' (after test) = 2
    assert result["punct_count"] == 2
    # No fully uppercase words
    assert result["upper_case_count"] == 0
    # "Hello" and "This" are title-case
    assert result["title_case_count"] == 2


def test_all_feature_keys():
    """Verify all 8 expected feature keys are present."""
    result = extract_stylometric_features("Some sample text.")
    expected_keys = {
        "char_count",
        "word_count",
        "avg_word_len",
        "punct_count",
        "sentence_count",
        "avg_sentence_len",
        "upper_case_count",
        "title_case_count",
    }
    assert set(result.keys()) == expected_keys


def test_longer_text():
    """Feature extraction on a multi-sentence paragraph."""
    text = (
        "Artificial intelligence is transforming the world. "
        "Many people are excited about the possibilities. "
        "However, some are concerned about the risks."
    )
    result = extract_stylometric_features(text)

    assert result["char_count"] > 0
    assert result["word_count"] > 10
    assert result["sentence_count"] == 3
    assert result["avg_word_len"] > 0
    assert result["avg_sentence_len"] > 0
    assert isinstance(result["avg_word_len"], float)

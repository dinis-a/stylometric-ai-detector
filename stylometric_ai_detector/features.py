"""Stylometric feature extraction for text analysis."""

from __future__ import annotations

import re
import string
from typing import Any, Dict, Optional, Union


def extract_stylometric_features(text: Optional[Union[str, Any]]) -> Dict[str, float]:
    """Extract 8 stylometric features from a text string.

    Features extracted:
        - char_count: Total number of characters
        - word_count: Total number of words
        - avg_word_len: Average word length
        - punct_count: Number of punctuation characters
        - sentence_count: Number of sentences
        - avg_sentence_len: Average sentence length in words
        - upper_case_count: Number of fully uppercase alphabetic words
        - title_case_count: Number of title-case words

    Args:
        text: Input text string. Non-string or empty values return all zeros.

    Returns:
        dict with keys: char_count, word_count, avg_word_len, punct_count,
                        sentence_count, avg_sentence_len, upper_case_count,
                        title_case_count.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "avg_word_len": 0.0,
            "punct_count": 0,
            "sentence_count": 0,
            "avg_sentence_len": 0.0,
            "upper_case_count": 0,
            "title_case_count": 0,
        }

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    punct_count = sum(1 for c in text if c in string.punctuation)

    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([s for s in sentences if s.strip()])
    sentence_count = sentence_count if sentence_count > 0 else 1  # avoid division by zero
    avg_sentence_len = word_count / sentence_count

    upper_case_count = sum(1 for w in words if w.isupper() and w.isalpha())
    title_case_count = sum(1 for w in words if w.istitle())

    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "punct_count": punct_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "upper_case_count": upper_case_count,
        "title_case_count": title_case_count,
    }

# stylometric-ai-detector

AI vs Human text detection using stylometric features and a Random Forest classifier.

## Overview

This library provides two functions:
- **`extract_stylometric_features(text)`** — Extract 8 stylometric features from any text.
- **`predict(text)`** — Classify text as AI-generated or human-written, with a confidence score.

The model is a Random Forest classifier achieving **96% accuracy**, trained on the [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) dataset. On first use, the model is automatically downloaded from Hugging Face and cached locally.

## Installation

```bash
pip install stylometric-ai-detector
```

## Quick Start

```python
from stylometric_ai_detector import extract_stylometric_features, predict

# Extract stylometric features
features = extract_stylometric_features("The quick brown fox jumps over the lazy dog.")
print(features)
# {'char_count': 44, 'word_count': 9, 'avg_word_len': 4.0, 'punct_count': 1, 'sentence_count': 1, 'avg_sentence_len': 9.0, 'upper_case_count': 0, 'title_case_count': 1}

# Predict AI vs Human (model auto-downloaded from Hugging Face on first call)
result = predict(text="Artificial intelligence is transforming our world.")
print(result)
# {'label': 'AI', 'probability': 0.99}

# Or pass pre-computed features
result = predict(features=features)
print(result)
# {'label': 'AI', 'probability': 0.91}
```

## Stylometric Features

| Feature            | Description                                   |
|--------------------|-----------------------------------------------|
| `char_count`       | Total number of characters                    |
| `word_count`       | Total number of words                         |
| `avg_word_len`     | Average word length                           |
| `punct_count`      | Number of punctuation characters              |
| `sentence_count`   | Number of sentences                           |
| `avg_sentence_len` | Average sentence length (in words)            |
| `upper_case_count` | Number of fully uppercase alphabetic words     |
| `title_case_count` | Number of title-case words                    |

## Model

The trained Random Forest model is hosted on Hugging Face at [`dinisds/stylometric-ai-detector`](https://huggingface.co/dinisds/stylometric-ai-detector). It is downloaded and cached to `~/.cache/stylometric-ai-detector/` on first use — no extra setup needed.

## Dataset

Trained on [Shanegerami's AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) dataset from Kaggle. The dataset contains ~487k text samples labeled as human-written (0) or AI-generated (1).

## License

MIT

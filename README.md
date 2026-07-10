# stylometric-ai-detector

[![PyPI version](https://img.shields.io/pypi/v/stylometric-ai-detector)](https://pypi.org/project/stylometric-ai-detector/)
[![Python](https://img.shields.io/pypi/pyversions/stylometric-ai-detector)](https://pypi.org/project/stylometric-ai-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> ⚠️ **This model is a baseline.** It was developed as a reference point for comparing more advanced (neural network-based) approaches from a separate project. While it achieves 96% accuracy, it relies on surface-level stylometric features and may not generalize across domains, languages, or AI model generations. Use as a benchmark, not as a final detector.

---

AI vs Human text detection using stylometric features and a Random Forest classifier. The package provides two simple functions — one to extract 8 stylometric features from any text, and one to classify text as AI-generated or human-written with a confidence score.

## Installation

```bash
pip install stylometric-ai-detector
```

## Quick Start

```python
from stylometric_ai_detector import extract_stylometric_features, predict

# Extract stylometric features from any text
features = extract_stylometric_features("The quick brown fox jumps over the lazy dog.")
# {'char_count': 44, 'word_count': 9, 'avg_word_len': 4.0, ...}

# Classify text — model auto-downloaded from Hugging Face on first call
result = predict(text="Artificial intelligence is transforming our world.")
# {'label': 'AI', 'probability': 0.99}

# Or pass pre-computed features
result = predict(features=features)
# {'label': 'AI', 'probability': 0.91}
```

## How It Works

The detector extracts **8 stylometric features** that capture surface-level patterns in text — character counts, word lengths, punctuation density, sentence structure, and capitalization. These features are fed into a Random Forest classifier trained on ~487k labeled samples.

| Feature              | Description                                    |
|----------------------|------------------------------------------------|
| `char_count`         | Total number of characters                     |
| `word_count`         | Total number of words                          |
| `avg_word_len`       | Average word length                            |
| `punct_count`        | Number of punctuation characters               |
| `sentence_count`     | Number of sentences                            |
| `avg_sentence_len`   | Average sentence length in words               |
| `upper_case_count`   | Number of fully uppercase alphabetic words     |
| `title_case_count`   | Number of title-case words                     |

## Model

The trained Random Forest model is hosted on Hugging Face at [`dinisds/stylometric-ai-detector`](https://huggingface.co/dinisds/stylometric-ai-detector).

On first use, the model is automatically downloaded and cached to `~/.cache/stylometric-ai-detector/` — no extra setup needed. You can override the repo by setting the `HF_MODEL_REPO` environment variable.

| Metric    | Value  |
|-----------|--------|
| Algorithm | Random Forest (100 estimators) |
| Accuracy  | 96.03% |
| Train set | 389,788 samples |
| Test set  | 97,447 samples |
| Dataset   | [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) |

## Development

```bash
# Clone and set up
git clone https://github.com/dinis-a/stylometric-ai-detector.git
cd stylometric-ai-detector
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Run tests
pytest -v

# Format code
isort . && black .

# Lint
isort --check-only --diff . && black --check --diff .
```

## Project Structure

```
├── stylometric_ai_detector/   # Library package
│   ├── __init__.py            # Public API: extract_stylometric_features, predict
│   ├── features.py            # Feature extraction logic
│   ├── predict.py             # Model loading & prediction
│   └── data/                  # Bundled model file
├── tests/                     # Test suite (pytest)
│   ├── test_features.py
│   └── test_predict.py
└── pyproject.toml             # Package metadata & tool config
```

## Limitations

- Trained on a single English-language dataset — may not generalize to other languages or domains
- Stylometric patterns vary across AI models and versions (this was trained on pre-2024 data)
- Relies on surface-level text statistics, not semantic understanding
- Intended as a **baseline** for comparison; production use-cases should prefer more robust approaches

## License

MIT — see [LICENSE](LICENSE) for details.

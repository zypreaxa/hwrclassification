# hwrclassification

[![PyPI - Version](https://img.shields.io/pypi/v/hwrclassification.svg)](https://pypi.org/project/hwrclassification)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hwrclassification.svg)](https://pypi.org/project/hwrclassification)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)
- [Data-collection](#data-collection)
- [NLP-functionality](#nlp-functionality)

## Installation

```console
pip install hwrclassification
```

## License

`hwrclassification` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Data collection
```
!wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip
```
```
!unzip -qq IAM_Words.zip
```
```
!sudo apt install unzip
```
```
!mkdir data
```
```
!mkdir data/words
```
```
!tar -xf IAM_Words/words.tgz -C data/words
```
```
!mv IAM_Words/words.txt data
```

## NLP functionality

Added NLP functionality:
- Exporting the transcribed words into texts (export_text_groups.py).
- Named entity recognition (NER.py)
- Text summary generation (summary.py)
- Keyword extraction (keyword_extraction.py)

Additional installations used:
- pip install spacy
- pip install "spacy[transformers]"
- python -m spacy download en_core_web_trf
- pip install tf-keras
- pip install tqdm
- pip install keybert sentence-transformers
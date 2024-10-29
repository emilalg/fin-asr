# Finnish Speech Recognition using LSTM-CTC

A deep learning model for Finnish speech recognition using LSTM networks with CTC (Connectionist Temporal Classification) loss.

## Overview

This project implements an LSTM-based speech recognition system specifically trained for Finnish language audio. It utilizes both Kielipankki and Hugging Face datasets for training and evaluation.

## Prerequisites

- Python 3.12.7+

## Install dependencies
```
pip install -r requirements.txt
```

exact versions used in training and running locally defined in freeze.txt

### Directory Structure
```
data/
├── kielipankki/
│   ├── dev-test/
│   ├── set1-part1/
│   └── set1-part2/
└── hf/
    └── [huggingface_dataset (common_voice_17)]
```

### Kielipankki Data
For all datasets except `dev-test`, place the inner folder from each part directly into the `kielipankki` directory.

### Hugging Face Data
If not pre-downloaded, you'll need to provide your Hugging Face token during runtime.


### Basic Training

```bash
python train.py \
    --data /path/to/data/folder \
    --lr 0.001 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4
```

### Using Pre-downloaded Data

```bash
python train.py \
    --data /path/to/data/folder \
    --predl true \
    --batch_size 32
```

### Using Hugging Face Datasets

```bash
python train.py \
    --data /path/to/data/folder \
    --token YOUR_HUGGINGFACE_TOKEN \
    --batch_size 32
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to data directory | Required |
| `--token` | Hugging Face API token | None |
| `--predl` | Use pre-downloaded data | False |
| `--lr` | Learning rate | 0.001 |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size for training | 32 |
| `--num_workers` | Number of data loading workers | 4 |

## Model Architecture

The model uses a bidirectional LSTM architecture with CTC loss for sequence-to-sequence learning.
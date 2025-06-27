# GFlowNet LLM Bayes Playground

This repository contains a minimal dataset generator, a small transformer model for the addition task and a stub demonstrating how one might start experimenting with a GFlowNet-style sampler.

## Installation

The code requires Python 3.8+ and PyTorch. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset generation

Run `python dataset.py` to generate `addition_dataset.pkl`. The file stores a
pickled list of `(input, output)` strings such as `"3 + 4"` and its result.

## Training the transformer

`python model.py` will train a tiny seq2seq transformer on the dataset. Training runs for a few epochs and prints the loss. The model architecture is simplified so it can run on a GPU such as the 3080.

## GFlowNet stub

`python gflownet_simple.py` demonstrates how to sample token sequences using the trained model. This is **not** a full GFlowNet implementation but should help start experimenting with the idea of sampling actions and computing probabilities.

## Testing the model

Run `python test_model.py` to train the transformer and then compute token-level accuracy on the dataset. The script also prints a few example generations so you can quickly gauge whether the model learned to add correctly.

Feel free to modify these scripts to explore Bayesian updates using GFlowNet sampling strategies.

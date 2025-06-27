# GFlowNet LLM Bayes Playground

This repository contains a minimal dataset generator, a small transformer model for the addition task and a stub demonstrating how one might start experimenting with a GFlowNet-style sampler.

## Dataset generation

Run `python dataset.py` to generate `addition_dataset.pkl` which contains pairs of inputs of the form `"a + b"` and their sum.

## Training the transformer

`python model.py` will train a tiny seq2seq transformer on the dataset. Training runs for a few epochs and prints the loss. The model architecture is simplified so it can run on a GPU such as the 3080.

## GFlowNet stub

`python gflownet_simple.py` demonstrates how to sample token sequences using the trained model. This is **not** a full GFlowNet implementation but should help start experimenting with the idea of sampling actions and computing probabilities.

Feel free to modify these scripts to explore Bayesian updates using GFlowNet sampling strategies.

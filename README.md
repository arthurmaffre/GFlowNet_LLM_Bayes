# Overview

This repository implements a novel research idea for enhancing Large Language Models (LLMs) with robust causal reasoning and Bayesian coherence using Generative Flow Networks (GFlowNets) as an adversarial component. The core concept is to redefine "intelligence" in AI systems as the ability to maintain Bayesian entropy—i.e., structured uncertainty and exploration—while ensuring internal consistency with an unstable environment.

Traditional LLMs are autoregressive predictors that estimate the next token $ P_\theta(x_{t+1} | x_{1:t}) $ based on prefixes, but they often fail to build explicit causal structures or adapt priors dynamically to evidence (e.g., updating disease probabilities based on contextual clues like travel history). This project addresses that by setting up an adversarial game:

- GFlowNet (Adversary): Generates batches of "perturbing" sequences that violate Bayesian relations (e.g., prior × likelihood ≠ posterior), acting as an "information adversary" to introduce non-sense and challenge the LLM.
- LLM (Defender): Reconstructs coherence by minimizing the divergence from Bayesian consistency, learning to internalize causal schemas robust to instability.

The approach is inspired by Bayesian principles and aims to create more human-like AI that avoids incoherent self-reinforcement loops. We start with a toy domain (simple additions) to test the idea empirically, with plans to scale to natural language.

Key Goals:
- Enforce $ P(\text{Prefix} | \text{Token}) \propto P(\text{Token} | \text{Prefix}) \times P(\text{Prefix}) $.
- Use entropy as a metric only under Bayesian constraints to promote useful diversity.
- Handle massive discrete spaces (e.g., $ 26^n $ for tokens) via GFlowNets' efficient sampling.

This is a prototype implementation.

## Motivation

From discussions (e.g., email thread with Prof. William J. Mccausland):

- Current LLMs predict tokens sequentially but lack explicit causal modeling. Example: "J’ai mal à la tête et le nez qui coule, quelle maladie ai-je ?" → High P(rhume), low P(tuberculose). But adding "Je suis allé en Inde" should boost P(tuberculose)—standard LLMs may not adapt priors well.
- Entropy alone can lead to useless diversity; it must be constrained by Bayesian coherence to avoid incoherent predictions (e.g., P(rain)=0.6 and P(no-rain)=0.6 sums >1).
- Problem: Huge search spaces make prior estimation impossible without flexible tools like GFlowNets.
- Solution: An unsupervised adversarial loop where GFlowNet creates adaptive, environment-grounded priors, and LLM enforces the Bayes equation. For both agents, we also compute gradients on the difference $ P(\text{Action}) \times P(\text{Info|Action}) - P(\text{sequence}) $ to preserve causal order over the entire sequence (not just next token). Here, P(Info|Action) is the product of token generation probabilities across the sequence for a given prefix, P(sequence) is the probability as if forcing GFlowNet to generate the full sequence (prefix + response), and the prior is GFlowNet's P if stopping at the "question" prefix.

This could lead to more robust AIs, closer to human knowledge internalization.

# Approach (V2: Bayesian GFlowNet for Causal LLMs)

## Key Definitions

- Sequence: $ x = [x_1, \dots, x_n] $, tokens as letters/digits.
- Prefix: Context window $ x_{t-L:t-1} $.
- LLM: Autoregressive $ x_{t+1} \sim P_\theta(x | x_{1:t}) $.
- Bayesian Coherence: Posterior ≈ prior × likelihood (normalized).
- Divergence Metric: $ |P(\text{Prefix} | \text{Token}) - P(\text{Token} | \text{Prefix}) \times P(\text{Prefix}) / Z| $, plus coherence term $ |P(\text{Action}) \times P(\text{Info|Action}) - P(\text{sequence})| $ over full sequences.

## Adversarial Game

1. GFlowNet Role:
    - Generates prefixes to maximize divergence, creating "non-sens" batches.
    - Reward: Divergence + causal order preservation (e.g., numbers before '+' in additions).
    - Adaptive Priors: Dynamically recalibrated from environment observations (e.g., token frequencies).
    - Training: Trajectory Balance (TB) loss for proportional sampling, with gradients on the coherence difference $ P(\text{Action}) \times P(\text{Info|Action}) - P(\text{sequence}) $.
2. LLM Role:
    - Minimizes the same divergence on perturbed batches, enforcing Bayes' equation.
    - Mixed Training: Blend generated (unsupervised, auto-compute targets if valid) and real data (supervised).
    - Loss: Cross-entropy + divergence penalty, with gradients on the coherence difference $ P(\text{Action}) \times P(\text{Info|Action}) - P(\text{sequence}) $.
3. Training Loop:
    - Alternate phases: GFlowNet challenges → LLM adapts.
    - Mix Ratio: Cursor (0-1) for generated vs. real data fraction—tune for efficiency.
    - Unsupervised Aspect: For generated sequences, parse and compute targets (e.g., sum for additions) if possible; skip invalid.
    - Sequence Optimization: Compute probs by multiplying across tokens; P(sequence) as GFlowNet-forced full dataset seq; prior as GFlowNet P up to prefix ("question").
4. Toy Domain: Additions
    - Inputs: "a + b =", Targets: "sum".
    - Environment: Rewards sequences preserving causal order (e.g., digit → '+' → digit → '=').

### Why GFlowNets?
- Efficient for discrete, high-dimensional spaces.
- Sample proportional to rewards (divergence), avoiding enumeration.
- Reference: https://arxiv.org/abs/2202.13903 (integrated for prior estimation).

## Installation

1. Clone the repo :

```bash
git clone https://github.com/arthurmaffre/GFlowNet_LLM_Bayes.git
cd GFlowNet_LLM_Bayes
```

2. Create a Conda environment (Python 3.10+)

```batch
conda create -n gflownet_llm python=3.13
conda activate gflownet_llm
````

3. Install dependencies:

```batch
pip install torch numpy matplotlib pandas pickle5
````

(Note: No internet access needed beyond this; code is self-contained.)

## Usage

The main script is test_model.py. It generates data, trains baseline LLM, runs adversarial training, and tests robustness.

**Note**: The code automatically detects and utilizes available hardware (CUDA, MPS for Apple Silicon, or CPU).

## Run the Script

```batch
python test_model.py
```

## Future Work

- Scale to text: Use QA datasets, perturb sentences for causal tests (e.g., disease diagnosis).
- Improve Priors: Integrate real environment estimation (e.g., token freq from corpus).
- Metrics: Add causal inference tests (e.g., do-calculus simulations).
- Hyperparams: Grid search for emb_dim, heads, epochs, mix_ratio.

## License

MIT-free to use/modify/cite.

Last Updated: July 10, 2025


## 🚧 Work in Progress

Tested ✅ / Untested ❌ / 🔷 not sure but has test


```batch
GFlowNet_LLM_Bayes/
├── README.md              # ✅ Project overview, motivation, etc.
├── requirements.txt       # ❌ Dependencies
├── constants.py           # ✅ Vocab, constants
├── dataset.py             # ✅ Data generation and loading
├── models.py              # 🔷 LLM and GFlowNet models
├── env.py                 # 🔷 Environment class
├── utils.py               # ❌ Helpers (sampling, losses, etc.)
├── train.py               # ❌ Training functions
├── test.py                # ❌ Testing functions
└── main.py                # ❌ Entry point to run everything
```
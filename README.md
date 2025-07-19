**⚠️ If you're not a fucking lunatic, get out of here.**

# Overview

The core thesis: true artificial intelligence is not merely about next-token prediction, it is about exploiting uncertainty to perform structured belief revision. In this view, stochasticity is not noise to be averaged out, but a latent structure, that must be constrained and shaped by Bayesian consistency.

Modern LLMs may achieve linguistic fluency, but they frequently violate fundamental probabilistic principles. They fail to update beliefs when presented with new evidence, and assign incoherent probabilities to mutually exclusive outcomes. This is not creativity, it is unstructured entropy masquerading as intelligence, a failure to enforce internal coherence in the face of uncertainty.

Traditional LLMs are autoregressive predictors that estimate the next token $ P_\theta(x_{t+1} | x_{1:t}) $ based on prefixes, but they often fail to build explicit causal structures (e.g., updating disease probabilities based on contextual clues like travel history). This project addresses that by setting up an adversarial game:

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

# Approach (V3: Bayesian GFlowNet for Causal LLMs via Differential Topology)

## Part1: Semantic Manifold Setup (Intuition: Map of Meanings)

View sequences as paths on a Riemannian manifold $\mathcal{M}$, points $p \in \mathcal{M}$ as embedded prefixes (vectors in $\mathbb{R}^d$). Metric $g$ from KL/cos sim for "semantic distance." Topology: Open sets around coherent prefixes. Intuition: Curved space where close points share causal logic; far ones diverge wildly. Differential structure for local flow, tangent spaces $T_p \mathcal{M}$ handle next-token perturbation.

## Part2: Transition Kernels and Flows (Intuition: Semantic Rivers)

Kernels $K_p : \mathcal{V} \rightarrow [0, 1]$ (LLM probs) as vector fields $X : \mathcal{M} \rightarrow T \mathcal{M}$, pushing geodesics. GFlowNet generates trajectories $\tau$ proportional to reward $R$ (divergence + causal order), inducing probability current $J = \rho v$ (Folker-Plank: )

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

**Note**: The code automatically detects and utilizes available hardware (CUDA > MPS (Apple) > CPU).

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

License
This project is licensed under the MIT License with an additional attribution requirement. You are free to use, modify, and distribute the code, provided that you include proper attribution to the original author in any derivative works, publications, or presentations that use or reference this code. Specifically:

Retain the copyright notice and this license in all copies or substantial portions of the software.
Cite this repository in any academic or technical publications as follows:

```LATEX
@misc{maffre2025gflownetllmbayes,
  author = {Arthur Maffre},
  title = {GFlowNet_LLM_Bayes: Enhancing LLMs with Causal Reasoning and Bayesian Coherence using GFlowNets},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/arthurmaffre/GFlowNet_LLM_Bayes}},
}
```
Failure to provide attribution may violate the terms of this license. See the LICENSE file for full details.



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
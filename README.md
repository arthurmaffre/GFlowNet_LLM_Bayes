**⚠️ If you're not a fucking lunatic, get out of here.**

**Project Level 🚀: Level 2 / 5** — [see the 5‑level scale here](https://github.com/arthurmaffre/arthurmaffre)

# LUCIDE : Latent Unified Causal Inference through Dynamic Equilibrium

For mathematical definitions and formal framework, see [definition.md](definition.md)

# Overview

As I argue:

> True artificial intelligence is not merely about next-token prediction, it is about exploiting uncertainty to perform structured belief revision. In this view, stochasticity is not noise to be averaged out, but a latent structure, that must be constrained and shaped by Bayesian consistency.

We propose a new architecture called **LUCIDE**: **Latent Unified Causal Inference through Dynamic Equilibrium**. LUCIDE is a 4-phase generative Bayesian model, based on the dynamic equilibrium between 4 entities:
- an environment prior $p^{env}_\theta$ (learned frequency of events)
- an internal prior $p^{internal}_\psi$ (prior of the learned causal model)
- a conditional Seq2Seq model $p^{LLM}_\phi(y|x)$
- an adversary that learns a contrastive distribution over contexts $p^{adv}_\omega(x)$.

The objective is to unify causal inference and generation through a self-regulated flow of probabilities.

# Motivation

Modern LLMs may achieve linguistic fluency, but they frequently violate fundamental causal principles. They fail to update beliefs when presented with new evidence, and assign incoherent causal inference to mutually exclusive outcomes. This is not creativity, it is unstructured entropy masquerading as intelligence, a failure to enforce internal coherence in the face of uncertainty.

Traditional LLMs are autoregressive predictors that estimate the next token $P_\phi^{LLM}(x_{t+1} | x_{1:t})$ based on prefixes, but they often fail to build explicit causal structures. For instance, consider a medical LLM: when a patient asks "I have a runny nose, what illness do I have?", the model might output $P(\text{cold} | \text{context}) \approx 0.9$ and $P(\text{tuberculosis} | \text{context}) \approx 0.01$. However, if the patient adds "I recently traveled to India" updated context defined as $context'$, the model should revise its beliefs—yet $P(\text{tuberculosis} | \text{context'})$ often remains near 0.01, failing to account for the fact that $P(\text{evidence traveled to india} | \text{tuberculosis})$ carries significant evidential weight in a proper causal model (India has higher tuberculosis prevalence than North America, where it is largely eradicated). A true causal reasoner would invoke Bayesian inversion and update accordingly. 

The prior of our belief system cannot rely solely on frequency of occurrence. Consider the example: "54234 + 13352" has virtually no chance of appearing in the LLM's training data, while "it's nice weather today" would appear significantly more often. We introduce the relation: $p_{\theta,\psi}^{\text{prior}} \propto p^{env}_\theta \times p^{internal}_\psi$.

Where:

- $p_{\theta,\psi}^{\text{prior}}$: the prior of our Bayesian model parametrized by $\theta, \psi$.
- $p^{env}_\theta$: the distribution of occurrence in the environment (observational frequency) parametrized by $\theta$.
- $p^{internal}_\psi$: the prior over our internal belief system (structural necessity) parametrized by $\psi$.

To visualize this, consider these examples:

- **"54234 + 13352"**: Here $p^{env}_\theta$ is low because we almost never observe this exact expression, but $p^{internal}_\psi \approx 1$ because if this were false, it would violate our entire mathematical belief system within a given formal framework.
- **"it's nice weather today"**: Here $p^{env}_\theta$ may be higher because small talk about weather is common, but $p^{internal}_\psi$ is lower because there is no strong causal necessity to this statement—it carries little inferential weight.

This approach is conceptually aligned with the **Integrated World Modeling Theory (IWMT)** framework from constructivist theories of consciousness, which posit that conscious experience arises from Bayesian inference over separately maintained internal models and external world distributions. In IWMT, the brain maintains distinct generative models: one representing the causal structure of the world, and another encoding the agent's internal beliefs and goals. Our decomposition $p_{\theta,\psi}^{\text{prior}} \propto p^{env}_\theta \times p^{internal}_\psi$ mirrors this distinction—separating observational frequency ($p^{env}_\theta$) from internal causal structure model in the agent's belief system ($p^{internal}_\psi$).

However, learning these distributions over the space of possible contexts cannot be achieved through classical sampling methods. We propose to use **GFlowNets** and **distributional reinforcement learning** to learn and infer these distributions in a tractable manner.

# Method

This project addresses this limitation by setting up un jeu en 4 phases:

$$p_{\theta,\psi}^{\text{prior}} = \frac{p^{env}_\theta \times p^{internal}_\psi}{Z_{\theta,\psi}}$$

## loop 1 (mise à jour sur l'environnement)

#### But : mettre à jour la distribution de prédiction ainsi que la distribution de fréquence d'apparition basée sur l'environnement

### Optimal 1 : &nbsp; ( $p^{\text{env}}_\theta$ )

$$p^{env}_\theta(x) \propto p^{env}(x), \quad \forall x \sim p^{env}$$

$$\qquad \text{where} \quad \theta^* = \arg \min_\theta \mathbb{E}_{x \sim p^{env}} \left[ \log \frac{p^{\text{env}}(x)}{p^{env}_\theta(x)}\right]$$

### Optimal 2 : &nbsp; ( $p_\phi(y|x)$ )

$$p_{\phi}^{\text{LLM}}(y|x) \propto p(y|x), \quad \forall (x,y) \sim p^{env}(x,y)$$

where $(x,y)$ are input-output pairs sampled from the environment distribution.

Equivalent, we minimize:

$$\phi^* = \arg \min_\phi \mathbb{E}_{(x,y) \sim p^{\text{env}}} \left[ - \log p_{\phi}^{\text{LLM}}(y|x) \right]$$

## loop 2 (mettre à jour le système de croyance interne façe aux nouvelles évidences)

#### But : mettre à jour son système de croyance interne avec notre distribution prédictive et celle de l'environnement mise à jour (phase de rêve ou on explore son propre système de croyance basés sur nos observations de l'environnement)

### Optimal 3

$$\underbrace{p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi}_{p_{\theta,\psi}^{\text{prior}}(x)} \times p_{\phi}^{\text{LLM}}(y|x) \propto p^{\text{env}}(x|y), \quad \forall x \sim p^{\text{internal}}$$

We minimize :

$$\psi^* = \arg \min_\psi \mathbb{E}_{x \sim p^{internal}_\psi} \left[ \log \left(\frac{Z_\psi^{\text{internal}} \times p_\psi^{\text{internal}}(x)}{R(x)}\right)^2 \right] \quad \text{with,} \quad R(x)=\underbrace{p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi(x)}_{p_{\theta,\psi}^{\text{prior}}(x)} \times p_\phi^{\text{LLM}}(y |x)$$

## loop 3

### Optimal 4

On cherche $p_{adv}$ qui va maximiser la divergence bayésienne

## loop 4

### Optimal 5

On va ajuster nos actions en se basant sur nos propres contradictions

## Phase 1

Sur le dataset de l'environnement

The approach is inspired by Bayesian principles and aims to create more human-like AI that avoids incoherent self-reinforcement loops. We start with a toy domain (simple additions) to test the idea empirically, with plans to scale to natural language.

Key Goals:
- Enforce $ P(\text{Prefix} | \text{Token}) \propto P(\text{Token} | \text{Prefix}) \times P(\text{Prefix}) $.
- Use entropy as a metric only under Bayesian constraints to promote useful diversity.
- Handle massive discrete spaces (e.g., $ 26^n $ for tokens) via GFlowNets' efficient sampling.

This is a prototype implementation.



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

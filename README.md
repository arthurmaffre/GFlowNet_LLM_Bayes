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

# The Frequency Paradox of Modern LLMs

## The Core Problem

Modern LLMs achieve linguistic fluency through pattern recognition, but systematically fail at higher-order causal reasoning. This isn't a simple scaling problem - it's a fundamental frequency issue in the data itself.

## The Frequency Hierarchy

At the lowest level, word combinations and grammatical structures appear millions of times in any dataset. The model easily learns these atomic building blocks. But as we stack these blocks into higher-order structures - from phrases to paragraphs to causal arguments - each specific configuration becomes exponentially rarer. 

- **Low-level patterns**: A particular grammatical pattern might appear millions of times
- **High-level patterns**: A specific chain of causal reasoning might appear only once, or never

The model is forced to generalize from increasingly sparse examples. When constructing complex logical arguments or mathematical proofs, it's not truly reasoning - it's attempting to pattern-match against configurations it has rarely or never seen. 

This is why LLMs often fail catastrophically in mathematics when writing expressions: each specific arrangement of symbols and logical steps is unique, and the model cannot rely on frequency-based pattern matching.

## The Scaling Illusion

No amount of data can solve this: the space of possible high-level conceptual "Lego towers" is virtually infinite, and each specific tower is essentially unique. The model succeeds at language because language's low-level patterns are redundant and frequent. It fails at reasoning because reasoning's high-level patterns are rare and specific.

## Failure of Causal Updates

Traditional LLMs are autoregressive predictors that estimate the next token $P_\phi^{LLM}(x_{t+1} | x_{1:t})$ based on prefixes, but they often fail to build explicit causal structures. 

### Medical Diagnosis Example

Consider a medical LLM:
1. **Initial query**: "I have a runny nose, what illness do I have?"
   - Model outputs: $P(\text{cold} | \text{context}) \approx 0.9$
   - Model outputs: $P(\text{tuberculosis} | \text{context}) \approx 0.01$

2. **Updated query**: Patient adds "I recently traveled to India" (context → context')
   - Expected: Model should revise beliefs using Bayesian updating
   - Reality: $P(\text{tuberculosis} | \text{context'})$ often remains near 0.01
   - Problem: Fails to account that $P(\text{India travel} | \text{tuberculosis})$ carries significant evidential weight

A true causal reasoner would invoke Bayesian inversion and update accordingly, recognizing that India has higher tuberculosis prevalence than North America, where it is largely eradicated.

## Beyond Frequency-Based Priors

The prior of our belief system cannot rely solely on frequency of occurrence. Consider: "54234 + 13352" has virtually no chance of appearing in the LLM's training data, while "it's nice weather today" would appear significantly more often.

### Proposed Framework

We introduce the relation:

$$p_{\theta,\psi}^{\text{prior}} \propto p_\theta^{env} \times p_\psi^{internal}$$

Where:
- $p_{\theta,\psi}^{\text{prior}}$: the prior of our Bayesian model parametrized by $\theta, \psi$
- $p^{env}_\theta$: the distribution of occurrence in the environment (observational frequency) parametrized by $\theta$
- $p^{internal}_\psi$: the prior over our internal belief system (structural necessity) parametrized by $\psi$

### Concrete Examples

| Expression | $p_\theta^{env}$ | $p_\psi^{internal}$ | Reasoning |
|------------|------------------|---------------------|-----------|
| **"54234 + 13352"** | ≈ 0 (never observed) | ≈ 1 (mathematically necessary) | If false, violates entire mathematical framework |
| **"it's nice weather today"** | High (common small talk) | Low (no causal necessity) | Carries little inferential weight |

This framework highlights the fundamental disconnect: LLMs optimize for $p_\theta^{env}$ while reasoning requires $p_\psi^{internal}$.

## Connection to Consciousness Theories

This approach is conceptually aligned with the **Integrated World Modeling Theory (IWMT)** framework from constructivist theories of consciousness, which posit that conscious experience arises from Bayesian inference over separately maintained internal models and external world distributions. In IWMT, the brain maintains distinct generative models: one representing the causal structure of the world, and another encoding the agent's internal beliefs and goals. 

Our decomposition $p_{\theta,\psi}^{\text{prior}} \propto p_\theta^{\text{env}} \times p_\psi^{internal}$ mirrors this distinction—separating observational frequency ($p_\theta^{\text{env}}$) from internal causal structure model in the agent's belief system ($p_\psi^{\text{internal}}$).

## Tractable Learning Through Modern Methods

However, learning these distributions over the space of possible contexts cannot be achieved through classical sampling methods. The combinatorial explosion of possible causal chains and belief updates makes traditional approaches intractable.

We propose to use **GFlowNets** and **distributional reinforcement learning** to learn and infer these distributions in a tractable manner:

- **GFlowNets**: Can learn to sample from complex compositional spaces, naturally handling the hierarchical structure of causal reasoning
- **Distributional RL**: Maintains full distributions over possible outcomes rather than point estimates, enabling proper uncertainty quantification

---

# Method: A Four-Phase Adversarial Bayesian Framework

To address these limitations, we propose a four-phase iterative framework that mirrors human cognitive processes: observation, internal reasoning, adversarial testing, and correction. The key insight is that instead of directly estimating intractable posteriors like $p(x|y)$, we leverage adversarial dynamics—much like humans refine their beliefs through internal debate and confrontation with counterexamples.

$$p_{\theta,\psi}^{\text{prior}} = \frac{p_\theta^{env} \times p_\psi^{internal}}{Z_{\theta,\psi}}$$

## Phase 1: Environmental Grounding
**Goal**: Update prediction and frequency distributions based on environmental observations

### Learning Environmental Frequency ($p^{\text{env}}_\theta$)

We align our environmental model with observed data:

$$p^{env}_\theta(x) \propto p^{env}(x), \quad \forall x \sim p^{env}$$

$$\theta^* = \arg \min_\theta \mathbb{E}_{x \sim p^{env}} \left[ \log \frac{p^{\text{env}}(x)}{p^{env}_\theta(x)}\right]$$

### Learning Conditional Generation ($p_\phi(y|x)$)

Standard autoregressive training on environmental data:

$$\phi^* = \arg \min_\phi \mathbb{E}_{(x,y) \sim p^{\text{env}}} \left[ - \log p_{\phi}^{\text{LLM}}(y|x) \right]$$

## Phase 2: Internal Belief Consolidation
**Goal**: Update internal belief system using the learned predictive and environmental distributions (analogous to dream-phase exploration of belief structures)

We seek consistency between our prior and posterior beliefs:

$$\underbrace{p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi}_{p_{\theta,\psi}^{\text{prior}}(x)} \times p_{\phi}^{\text{LLM}}(y|x) \propto p^{\text{env}}(x|y), \quad \forall x \sim p^{\text{internal}}$$

Optimization objective:

$$\psi^* = \arg \min_\psi \mathbb{E}_{x \sim p^{internal}_\psi} \left[  \left(\log \frac{Z_\psi^{\text{internal}} \times p_\psi^{\text{internal}}(x)}{R(x)}\right)^2 \right]$$

where $R(x) = p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi(x) \times p_\phi^{\text{LLM}}(y|x)$

## Phase 3: Adversarial Exploration
**Goal**: Discover sequences that violate Bayesian coherence—finding the blind spots in our reasoning

We learn an adversarial distribution $p_\omega^\text{adv}$ that generates contexts where our model fails Bayesian consistency:

$$\underbrace{p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi}_{p_{\theta,\psi}^{\text{prior}}(x)} \times p_{\phi}^{\text{LLM}}(y|x) \not\propto p^{\text{env}}(x|y), \quad \forall x \sim p^{\text{adv}}_\omega$$

We maximize Bayesian divergence:

$$\omega^* = \arg \max_\omega \mathbb{E}_{x \sim p^{adv}_\omega} \left[ \left(\log \frac{Z_\omega^{\text{adv}} \times p_\omega^{\text{adv}}(x)}{R(x)}\right)^2 \right]$$

This mirrors human cognition: we actively seek counterexamples and edge cases that challenge our beliefs, forcing deeper understanding.

## Phase 4: Adversarial Correction
**Goal**: Restore Bayesian coherence on adversarial contexts—learning from our mistakes

We adjust the generative model to handle the discovered inconsistencies:

$$\phi^* = \arg \min_\phi \mathbb{E}_{x \sim p^{\text{adv}}_\omega} \left[ - \log \left( p^{\text{env}}_\theta(x) \times p^{\text{internal}}_\psi(x) \times p_\phi^{\text{LLM}}(y|x) \right) \right]$$

By focusing on adversarial examples, we maximize marginal likelihood $p(y)$ without explicitly computing the intractable posterior $p(x|y)$.

## Intelligence Metrics via ELBO

We can create a general intelligence metric using the Evidence Lower Bound:

$$\log p(y) \geq \text{ELBO} \approx MI(y, x) - H[y|x]$$

where:
- $MI(y, x)$: Mutual Information between inputs and outputs
- $H[y|x]$: Conditional entropy of outputs given inputs

### Connection to GFlowNets

Remarkably, GFlowNets provide exact tools for this framework. From GFlowNets foundations (p.39):

$$H[S|x] = \frac{F'(s_0 | x)}{F(s_0 | x)} + \log F(s_0 | x)$$

And the entropic reward function (Definition 53):

$$R'(s) = -R(s) \log R(s), \quad \text{where } 0 \leq R(s) < 1$$

This allows us to estimate mutual information through dual GFlowNet training:

$$MI(S; X) = H[S] - \mathbb{E}_X[H[S | X]] = \frac{F'(s_0)}{F(s_0)} + \log F(s_0) - \mathbb{E}_X \left[ \frac{F'(s_0 | X)}{F(s_0 | X)} + \log F(s_0 | X) \right]$$

The alignment between our adversarial Bayesian framework and GFlowNets' entropic formulation suggests a deep connection between causal reasoning, adversarial learning, and information-theoretic measures of intelligence.


# LUCIDE: Test 1 — Validation on Addition Dataset

To empirically validate the LUCIDE framework, we conduct an initial experiment using a synthetic addition dataset. This test assesses the model's ability to perform causal inference and generalization in a controlled arithmetic domain, where ground-truth causal relationships (i.e., addition rules) are well-defined and verifiable. The experiment focuses on evaluating generalization to unseen numbers, comparing LUCIDE against a baseline autoregressive LLM trained with teacher forcing. This serves as a foundational step before scaling to more complex text corpora, such as causal reasoning in natural language.

## Dataset Description
We utilize the addition dataset generated by the provided script (`dataset.py`), which produces exhaustive pairs of additions ```a+b=c``` where $a, b \in [0, 99]$. The dataset is shuffled with seed = 42 but is split deterministically:

- **Training set**: All pairs except those where both $a$ and $b$ are in $[40, 49]$ ($9,900$ samples).
- **Evaluation set**: Pairs where both $a$ and $b$ are in $[40, 49]$ (100 samples).

Each sample is formatted as a sequence pair: input string (e.g., ```43+66=```) and target string (e.g., ```109```). Metadata includes vocabulary (digits 0-9, '+', '='), maximum sequence lengths, and other parameters for consistency.

## Experimental Setup
### Baseline: Standard LLM with Teacher Forcing
**Model**: A compact autoregressive sequence-to-sequence model (e.g., based on GRU architecture with 2 layers, 64 hidden dimensions).

**Training**: Fine-tuned on the training set using teacher forcing, optimizing cross-entropy loss: 

$$
\alpha^* = \arg \min_\alpha \mathbb{E}_{(x,y)} \left[ - \log p_\alpha (y|x) \right]
$$

**Evaluation**: Measure accuracy on the held-out evaluation set (additions in [40,49]).

### LUCIDE
**Model**: Same base architecture as baseline (GRU, 2 layers, 64 hidden dimensions), but embedded within the four-component LUCIDE system.

**Training**: Iterative four-phase training over 5 complete cycles:
- **Phase 1**: Align $p^{env}_\theta$ and $p^{LLM}_\phi$ with training data
- **Phase 2**: Optimize $p^{internal}_\psi$ using GFlowNets for structural consistency
- **Phase 3**: Generate adversarial samples via $p^{adv}_\omega$ using distributional RL
- **Phase 4**: Fine-tune $p^{LLM}_\phi$ on adversarial samples for coherence restoration

**Evaluation**: Same evaluation protocol as baseline, with additional ELBO-based intelligence metric tracking.


## Experimental Hypotheses

### Baseline Expectations
Strong performance on in-distribution patterns but significant degradation on out-of-distribution inputs. Specifically, we anticipate <50% accuracy on the [40,49] test range due to over-reliance on superficial frequency patterns.

### LUCIDE Predictions
Substantial improvement in OOD generalization (targeting >80% accuracy), achieved through:
- Structural priors that encode arithmetic necessity
- Adversarial training that addresses frequency-based biases
- Enhanced uncertainty quantification via the intelligence metric

### Success Criteria
LUCIDE demonstrates meaningful advantage if:
1. Generalization gap improves with statistical significance (p < 0.05, bootstrap resampling)
2. ELBO metric shows superior information efficiency
3. Qualitative analysis reveals structurally sound reasoning on edge cases

---

## Broader Impact

This test serves as a proof-of-concept for LUCIDE's core thesis: that meaningful intelligence requires integrating environmental statistics with structural priors through adversarial refinement. Success here validates the framework's potential for extension to complex domains including:

- Causal reasoning in medical literature
- Logical inference chains
- Multi-step problem decomposition

The simplicity of addition ensures interpretability while capturing fundamental challenges in generalization—making it an ideal starting point for validating LUCIDE's principles before scaling to natural language tasks.

## License

License
This project is licensed under the MIT License with an additional attribution requirement. You are free to use, modify, and distribute the code, provided that you include proper attribution to the original author in any derivative works, publications, or presentations that use or reference this code. Specifically:

Retain the copyright notice and this license in all copies or substantial portions of the software.
Cite this repository in any academic or technical publications as follows:

```LATEX
@misc{LUCIDE,
  author = {Arthur Maffre},
  title = {LUCIDE: Latent Unified Causal Inference through Dynamic Equilibrium},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/arthurmaffre/LUCIDE}},
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

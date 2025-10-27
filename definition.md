# LUCIDE: Mathematical Definitions and Framework
## Latent Unified Causal Inference through Dynamic Equilibrium

---

## 1. Core Probability Spaces

### 1.1 Basic Spaces

Let $\mathcal{V}$ be the finite vocabulary (tokens) with $|\mathcal{V}| = V$.

Let $\mathcal{S} = \mathcal{V}^*$ be the space of all finite sequences over $\mathcal{V}$.

Let $\mathcal{S}_n = \mathcal{V}^n$ be the space of sequences of length exactly $n$.

### 1.2 Probability Measures

We define four fundamental probability measures on $\mathcal{S}$:

1. **Environmental Prior**: $p_{\text{env}}: \mathcal{S} \to [0,1]$
   - Represents observational frequency in training/environment data
   - Empirically estimated from corpus: $p_{\text{env}}(s) = \frac{\text{count}(s)}{\sum_{s' \in \mathcal{S}} \text{count}(s')}$

2. **Internal Prior**: $p_{\text{internal}}: \mathcal{S} \to [0,1]$
   - Represents structural necessity within the belief system
   - High for logically/causally necessary sequences
   - Low for contingent/arbitrary sequences

3. **Unified Prior**: $p_{\text{prior}}: \mathcal{S} \to [0,1]$
   $$p_{\text{prior}}(s) = \frac{p_{\text{env}}(s) \cdot p_{\text{internal}}(s)}{Z}$$
   where $Z = \sum_{s \in \mathcal{S}} p_{\text{env}}(s) \cdot p_{\text{internal}}(s)$ is the normalization constant.

4. **Adversarial Distribution**: $p_{\text{adv}}: \mathcal{S} \to [0,1]$
   - Learned by GFlowNet to maximize divergence from LLM

---

## 2. Conditional Models

### 2.1 Language Model

The LLM is a conditional distribution:
$$p_{\text{LLM}}(y|x; \theta): \mathcal{S} \times \mathcal{S} \to [0,1]$$

where $x$ is the input/prefix and $y$ is the output/completion.

Autoregressive decomposition:
$$p_{\text{LLM}}(y|x; \theta) = \prod_{t=1}^{|y|} p_{\text{LLM}}(y_t | x, y_{1:t-1}; \theta)$$

### 2.2 GFlowNet Model

The GFlowNet learns a policy $\pi_{\phi}: \mathcal{S} \to \Delta(\mathcal{V})$ where $\Delta(\mathcal{V})$ is the probability simplex over tokens.

Flow matching objective:
$$\mathcal{L}_{\text{TB}}(\phi) = \mathbb{E}_{\tau \sim \pi_\phi}\left[\left(\log Z_\phi + \log P_F(\tau) - \log R(\tau) - \log P_B(\tau)\right)^2\right]$$

where:
- $P_F(\tau) = \prod_{t} \pi_\phi(a_t|s_t)$ is forward probability
- $P_B(\tau)$ is backward probability (uniform in our case)
- $R(\tau)$ is the reward function
- $Z_\phi$ is the learned partition function

---

## 3. Bayesian Coherence Constraints

### 3.1 Fundamental Equation

For any prefix $x$ and token $v \in \mathcal{V}$:

$$p(x|v) = \frac{p(v|x) \cdot p(x)}{p(v)}$$

where:
- $p(v|x) = p_{\text{LLM}}(v|x; \theta)$ (next token probability)
- $p(x) = p_{\text{prior}}(x)$ (prior over prefixes)
- $p(v) = \sum_{x' \in \mathcal{S}} p(v|x') \cdot p(x')$ (marginal)

### 3.2 Divergence Measure

The Bayesian divergence for a token $v$ given prefix $x$:

$$D_{\text{Bayes}}(v, x) = \left| p_{\text{posterior}}(x|v) - p_{\text{likelihood}}(v|x) \cdot p_{\text{prior}}(x) \right|$$

where $p_{\text{posterior}}(x|v) = \frac{p_{\text{LLM}}(v|x) \cdot p_{\text{prior}}(x)}{\sum_{x'} p_{\text{LLM}}(v|x') \cdot p_{\text{prior}}(x')}$

### 3.3 Causal Order Coherence

For a sequence $s = (s_1, \ldots, s_n)$, define:

$$C_{\text{causal}}(s) = \left| P(\text{Action}) \cdot P(\text{Info}|\text{Action}) - P(s) \right|$$

where:
- $P(\text{Action})$ = probability of generating the prefix/question
- $P(\text{Info}|\text{Action})$ = probability of generating the answer given the question
- $P(s)$ = joint probability of the entire sequence

---

## 4. Reward Structure

### 4.1 GFlowNet Reward

The reward for a trajectory $\tau$ ending in state $s$:

$$R(\tau) = R_{\text{base}}(s) + \lambda_1 \cdot D_{\text{Bayes}}(s) + \lambda_2 \cdot \mathbb{1}[\text{causal-order}(s)]$$

where:
- $R_{\text{base}}(s)$ = base environment reward (e.g., correctness for math)
- $D_{\text{Bayes}}(s)$ = Bayesian divergence from LLM
- $\mathbb{1}[\text{causal-order}(s)]$ = indicator for preserving causal structure
- $\lambda_1, \lambda_2 > 0$ are hyperparameters

### 4.2 LLM Loss

The LLM minimizes:

$$\mathcal{L}_{\text{LLM}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \alpha \cdot \mathcal{L}_{\text{div}}(\theta) + \beta \cdot \mathcal{L}_{\text{causal}}(\theta)$$

where:
- $\mathcal{L}_{\text{CE}}$ = cross-entropy loss on training data
- $\mathcal{L}_{\text{div}}$ = penalty for Bayesian divergence
- $\mathcal{L}_{\text{causal}}$ = penalty for violating causal coherence

---

## 5. Dynamic Equilibrium

### 5.1 Nash Equilibrium Formulation

The system seeks a Nash equilibrium $(\theta^*, \phi^*)$ where:

1. $\theta^* \in \arg\min_\theta \mathcal{L}_{\text{LLM}}(\theta, \phi^*)$
2. $\phi^* \in \arg\max_\phi \mathbb{E}_{\tau \sim \pi_\phi}[R(\tau; \theta^*)]$

### 5.2 Fixed Point Iteration

The training alternates between:

**Phase 1** (GFlowNet update):
$$\phi_{t+1} = \phi_t - \eta_\phi \nabla_\phi \mathcal{L}_{\text{TB}}(\phi_t; \theta_t)$$

**Phase 2** (LLM update):
$$\theta_{t+1} = \theta_t - \eta_\theta \nabla_\theta \mathcal{L}_{\text{LLM}}(\theta_t; \phi_t)$$

### 5.3 Convergence Criteria

Convergence is measured by:
1. **Divergence stability**: $\text{Var}[D_{\text{Bayes}}] < \epsilon_1$
2. **Causal consistency**: $\mathbb{E}[C_{\text{causal}}] < \epsilon_2$
3. **Performance plateau**: $|\mathcal{L}_{t} - \mathcal{L}_{t-k}| < \epsilon_3$

---

## 6. Semantic Manifold (Advanced)

### 6.1 Manifold Structure

Let $\mathcal{M}$ be a Riemannian manifold with:
- Points: embedded sequence representations $\varphi: \mathcal{S} \to \mathcal{M} \subset \mathbb{R}^d$
- Metric tensor: $g_{ij}(p) = \langle \frac{\partial \varphi}{\partial s_i}, \frac{\partial \varphi}{\partial s_j} \rangle$

### 6.2 Probability Flow

The probability current on $\mathcal{M}$:
$$J(p, t) = \rho(p, t) \cdot v(p, t)$$

where:
- $\rho(p, t)$ = probability density at point $p$ at time $t$
- $v(p, t)$ = velocity field induced by token transitions

Satisfies continuity equation:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot J = 0$$

### 6.3 Geodesic Distance

The semantic distance between sequences $s_1, s_2$:
$$d_{\mathcal{M}}(s_1, s_2) = \inf_{\gamma} \int_0^1 \sqrt{g(\dot{\gamma}(t), \dot{\gamma}(t))} dt$$

where $\gamma: [0,1] \to \mathcal{M}$ is a path connecting $\varphi(s_1)$ to $\varphi(s_2)$.

---

## 7. Implementation Mapping

### 7.1 Discrete to Continuous

The discrete token space maps to continuous representations:

$$\text{Embed}: \mathcal{V} \to \mathbb{R}^{d_{\text{emb}}}$$

with learned parameters $W_{\text{emb}} \in \mathbb{R}^{V \times d_{\text{emb}}}$.

### 7.2 Transformer Architecture

**LLM** (Seq2SeqTransformer):
- Encoder: $h^{\text{enc}} = \text{TransformerEncoder}(\text{Embed}(x) + \text{Pos}(x))$
- Decoder: $h^{\text{dec}} = \text{TransformerDecoder}(\text{Embed}(y) + \text{Pos}(y), h^{\text{enc}})$
- Output: $p(y_t|x, y_{<t}) = \text{Softmax}(W_{\text{out}} h^{\text{dec}}_t)$

**GFlowNet** (FlowNet):
- State encoding: $h = \text{TransformerEncoder}(\text{Embed}(s) + \text{Pos}(s))$
- Policy: $\pi(a|s) = \text{LogSoftmax}(W_{\text{out}} h_{-1})$ (last hidden state)

### 7.3 Practical Approximations

1. **Prior Estimation**:
   - $p_{\text{env}}$: Estimated from training corpus frequencies
   - $p_{\text{internal}}$: Approximated by rule-based heuristics or learned network

2. **Marginal Computation**:
   - Full marginalization intractable
   - Use Monte Carlo approximation: $p(v) \approx \frac{1}{N}\sum_{i=1}^N p(v|x_i) p(x_i)$

3. **Divergence Calculation**:
   - Compute on mini-batches
   - Use moving averages for stability

---

## 8. Training Algorithm

### 8.1 Initialization

```
θ₀ ← PretrainedLLM or RandomInit
φ₀ ← RandomInit
Z₀ ← 0 (log partition function)
```

### 8.2 Main Loop

```
for epoch = 1 to T:
    # Phase 1: GFlowNet maximizes divergence
    for batch in GFlowBatches:
        τ ~ π_φ (sample trajectories)
        R(τ) = R_base(τ) + λ₁·D_Bayes(τ,θ) + λ₂·C_causal(τ)
        L_TB = TB_Loss(τ, R, φ)
        φ ← φ - η_φ·∇_φ L_TB
    
    # Phase 2: LLM minimizes divergence
    for batch in MixedBatches:
        x_gen, y_gen ~ π_φ (generated data)
        x_real, y_real ~ Data (real data)
        x_mix = concat(x_gen, x_real) with ratio ρ
        
        L_CE = CrossEntropy(p_LLM(y|x;θ), y)
        L_div = D_Bayes(x,y,θ,φ)
        L_causal = C_causal(x,y,θ)
        
        L = L_CE + α·L_div + β·L_causal
        θ ← θ - η_θ·∇_θ L
```

### 8.3 Convergence Monitoring

Track metrics:
- $\text{Div}_t = \mathbb{E}[D_{\text{Bayes}}]$ (should stabilize)
- $\text{Acc}_t$ = task accuracy (should improve)
- $\text{Rob}_t$ = robustness to noise (should increase)

---

## 9. Theoretical Properties

### 9.1 Existence of Equilibrium

**Theorem 1** (Existence): Under mild conditions (compact parameter spaces, continuous losses), there exists at least one Nash equilibrium $(\theta^*, \phi^*)$.

**Proof sketch**: Apply Brouwer's fixed-point theorem to the best-response mapping.

### 9.2 Bayesian Consistency

**Theorem 2** (Consistency): At equilibrium, the system satisfies approximate Bayesian coherence:
$$\mathbb{E}_{x,v}\left[ \left| p(x|v) - \frac{p(v|x)p(x)}{p(v)} \right| \right] < \epsilon$$

**Proof sketch**: By construction of the loss functions and adversarial training.

### 9.3 Causal Invariance

**Proposition 1**: The learned model exhibits improved causal invariance:
$$\text{Var}_{\text{noise}}[p(y|x + \epsilon)] < \text{Var}_{\text{baseline}}[p(y|x + \epsilon)]$$

where $\epsilon$ represents input perturbations.

---

## 10. Connection to Physics

### 10.1 Thermodynamic Analogy

The system can be viewed as minimizing free energy:
$$F = U - TS$$

where:
- $U = -\mathbb{E}[\log p(y|x)]$ (internal energy / prediction loss)
- $S = -\sum p \log p$ (entropy)
- $T = \frac{1}{\beta}$ (temperature parameter controlling exploration)

### 10.2 Information Geometry

The Fisher information metric on parameter space:
$$g_{ij}(\theta) = \mathbb{E}\left[ \frac{\partial \log p(x;\theta)}{\partial \theta_i} \frac{\partial \log p(x;\theta)}{\partial \theta_j} \right]$$

Natural gradient updates follow geodesics in this space.

### 10.3 Hamiltonian Dynamics

The system evolution can be described by:
$$\begin{cases}
\frac{d\theta}{dt} = -\frac{\partial H}{\partial \phi} \\
\frac{d\phi}{dt} = \frac{\partial H}{\partial \theta}
\end{cases}$$

where $H(\theta, \phi) = \mathcal{L}_{\text{LLM}}(\theta) - \mathbb{E}[R(\tau; \phi)]$ is the Hamiltonian.

---

## 11. Key Innovations

1. **Factorized Prior**: $p_{\text{prior}} = p_{\text{env}} \times p_{\text{internal}}$ separates frequency from necessity
2. **Adversarial Causal Learning**: GFlowNet generates challenging cases for causal reasoning
3. **Bayesian Divergence Reward**: Enforces probabilistic coherence
4. **Unsupervised Adaptation**: Can learn from generated sequences without labels

---

## 12. Open Questions

1. **Scalability**: How to efficiently compute marginals for large vocabularies?
2. **Prior Learning**: How to learn $p_{\text{internal}}$ from data?
3. **Convergence Rate**: What determines the speed of convergence to equilibrium?
4. **Generalization**: Does the framework extend to multi-modal inputs?
5. **Interpretability**: Can we extract explicit causal graphs from the learned model?

---

## References

- Bengio et al. (2021): "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"
- Pearl (2009): "Causality: Models, Reasoning, and Inference"
- MacKay (2003): "Information Theory, Inference, and Learning Algorithms"
- Integrated World Modeling Theory (IWMT): Consciousness as Bayesian inference
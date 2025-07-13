import torch
import math
from torch.distributions import Categorical
import numpy as np
from constants import DEVICE, char2idx, BOS, EOS, VOCAB_SIZE, idx2char, MAX_LEN
from models import FlowNet

def sample_trajectory(gflow_model: FlowNet, env):
    gflow_model.eval()
    state = env.reset()
    trajectory = state[:]
    done = False
    reward = 0.0
    while not done:
        prefix = torch.tensor([state], device=DEVICE)
        logp = gflow_model(prefix)  # (1, VOCAB_SIZE)
        action = Categorical(logits=logp).sample().item()
        state, step_reward, done = env.step(action)
        trajectory.append(action)
        reward += step_reward
    return trajectory, reward

def tb_loss(gflow_model: FlowNet, trajectories, rewards):
    losses = []
    for tokens, R in zip(trajectories, rewards):
        logfwd = 0.0
        for k in range(len(tokens) - 1):
            prefix = torch.tensor([tokens[:k+1]], device=DEVICE)
            logp = gflow_model(prefix)
            logfwd += logp[0, tokens[k+1]]
        logbwd = -math.log(VOCAB_SIZE) * (len(tokens) - 1)
        tb = gflow_model.logZ + logfwd - math.log(max(R, 1e-8)) - logbwd
        losses.append(tb ** 2)
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=DEVICE)

def compute_bayesian_divergence(llm_model, prefix_tensor, token_tensor, prior_prob, gflow_model=None):
    llm_model.eval()
    with torch.no_grad():
        tgt_input = torch.tensor([[char2idx[BOS]]], device=DEVICE)
        logits = llm_model(prefix_tensor, tgt_input)[:, -1, :]
        likelihood = torch.softmax(logits, dim=-1)[0, token_tensor]
        p_token_approx = likelihood * prior_prob
        Z = max(p_token_approx, 1e-8)  # Simple norm
        posterior = (likelihood * prior_prob) / Z
        divergence = abs(posterior - (likelihood * prior_prob))
        # Coherence term: Use actual probs from GFlowNet if provided
        action_prob = 1.0
        info_prob = 1.0
        if gflow_model:
            seq_logp = 0.0
            for i in range(1, prefix_tensor.shape[1]):
                pref = prefix_tensor[:, :i]
                logp = gflow_model(pref)
                seq_logp += logp[0, prefix_tensor[0, i]]
            action_prob = torch.exp(seq_logp / prefix_tensor.shape[1])
            info_prob = likelihood  # Approx P(Info|Action)
        coherence_term = abs(action_prob * info_prob - prior_prob)
        return divergence + coherence_term

def generate(model, prompt_str, max_len=MAX_LEN):
    model.eval()
    src = torch.tensor([[char2idx[c] for c in prompt_str]], device=DEVICE)
    tgt = torch.tensor([[char2idx[BOS]]], device=DEVICE)
    generated = []
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(src, tgt)[:, -1, :]
            token = logits.argmax(-1).item()
        generated.append(token)
        tgt = torch.cat([tgt, torch.tensor([[token]], device=DEVICE)], dim=1)
        if token == char2idx[EOS]:
            break
    return ''.join(idx2char[t] for t in generated if t != char2idx[EOS])

def parse_and_compute_target(prefix_str):
    try:
        parts = prefix_str.split('+')
        a = int(parts[0].strip())
        b_part = parts[1].split('=')[0].strip()
        b = int(b_part)
        return str(a + b)
    except:
        return None
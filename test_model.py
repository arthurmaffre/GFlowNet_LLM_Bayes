import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions import Categorical
import random
import pickle
import numpy as np
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence



# Constants
from constants import VOCAB_SIZE, char2idx, idx2char, PAD, DEVICE, MAX_LEN, BOS, EOS

# Utils
from utils import print_number_params

print(f"\n✅ Model initialized on device: {DEVICE} (PyTorch)\n")


print(f"Vocab size: {VOCAB_SIZE}")

# Dataset Generation
def generate_addition_dataset(num_samples: int = 5000, max_val: int = 99) -> List[Tuple[str, str]]:
    data: List[Tuple[str, str]] = []
    for _ in range(num_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        result = a + b
        input_str = f"{a} + {b} ="
        target_str = str(result)
        data.append((input_str, target_str))
    return data

data = generate_addition_dataset()
with open("addition_dataset.pkl", "wb") as f:
    pickle.dump(data, f)
print(f"Generated {len(data)} samples. Example: {data[0]}")

# Load Dataset
with open("addition_dataset.pkl", "rb") as f:
    full_data = pickle.load(f)
train_data = full_data[:4000]
test_data = full_data[4000:]
print(f"Loaded {len(train_data)} train samples, {len(test_data)} test samples. \n")

# Seq2SeqTransformer (LLM)
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 16, n_heads: int = 2, n_layers: int = 1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(MAX_LEN + 1, emb_dim)
        self.dropout = nn.Dropout(0.1)

        enc_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dropout=0.1, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(emb_dim, n_heads, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.fc = nn.Linear(emb_dim, vocab_size)
        self.fc.weight = self.emb.weight

    def add_pos(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.emb(x) + self.pos(positions)

    @staticmethod
    def causal_mask(sz, device):
        # Changed to bool type: True where masked (do not attend), False where attendable.
        # This creates a upper triangular mask (future positions masked).
        return torch.triu(torch.ones((sz, sz), dtype=bool, device=device), diagonal=1)

    def forward(self, src, tgt):
        src_emb = self.dropout(self.add_pos(src))
        tgt_emb = self.dropout(self.add_pos(tgt))

        tgt_mask = self.causal_mask(tgt.size(1), tgt.device)
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)
        return self.fc(out)

# FlowNet (GFlowNet)
class FlowNet(nn.Module):
    def __init__(self, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=char2idx[PAD])
        self.pos = nn.Embedding(MAX_LEN + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Linear(d_model, VOCAB_SIZE)
        self.fc.weight = self.emb.weight
        self.logZ = nn.Parameter(torch.zeros(()))  # Learned normalizer

    def causal_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(DEVICE)

    def forward(self, prefix):
        B, L = prefix.shape
        pos = torch.arange(L, device=prefix.device).unsqueeze(0).expand(B, L)
        x = self.emb(prefix) + self.pos(pos)
        mask = self.causal_mask(L)
        x = self.tr(x, mask=mask)
        logits = self.fc(x[:, -1])  # Logits for next token (B, VOCAB_SIZE)
        return torch.log_softmax(logits, dim=-1)  # Log probs

# Environment
class AddSeqEnv:
    def __init__(self, max_len=MAX_LEN):
        self.max_len = max_len

    def reset(self):
        self.state = [char2idx[BOS]]
        return self.state

    def step(self, action):
        self.state.append(action)
        done = (action == char2idx[EOS]) or (len(self.state) >= self.max_len)
        reward = 0.0
        if done:
            seq_str = ''.join(idx2char[t] for t in self.state[1:-1] if t != char2idx[EOS])
            # Reward higher if causal order preserved (e.g., numbers before +)
            if self.preserves_causal_order(seq_str):
                reward = 1.0
            else:
                reward = 0.01
        return self.state, reward, done

    def preserves_causal_order(self, s):
        # Simple check: + appears after first number, before second
        if '+' not in s:
            return False
        parts = s.split('+')
        if len(parts) < 2:
            return False
        return parts[0].strip().isdigit() and '=' in parts[1]

env = AddSeqEnv()

# Helpers
def sample_trajectory(gflow_model, env):
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
        reward += step_reward  # Accumulate
    return trajectory, reward

def tb_loss(gflow_model, trajectories, rewards):
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
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)

def compute_bayesian_divergence(llm_model, prefix_tensor, token_tensor, prior_prob, action_prob=1.0, info_prob=1.0):
    llm_model.eval()
    with torch.no_grad():
        tgt_input = torch.tensor([[char2idx[BOS]]], device=DEVICE)
        logits = llm_model(prefix_tensor, tgt_input)[:, -1, :]
        likelihood = torch.softmax(logits, dim=-1)[0, token_tensor.item()]  # Fix: [0, index]
        p_token_approx = likelihood * prior_prob
        posterior = (likelihood * prior_prob) / max(p_token_approx, 1e-8)
        divergence = abs(posterior - (likelihood * prior_prob))
        # Add new coherence term: P(Action) * P(Info|Action) ≈ P(sequence)
        coherence_term = abs(action_prob * info_prob - prior_prob)
        return divergence + coherence_term  # Combined

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
        result = str(a + b)
        return result  # Just the sum
    except:
        return None  # Invalid, unsupervised skip

# Training
llm_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
# Train baseline first (simple CE on data)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
llm_opt = optim.Adam(llm_model.parameters(), lr=5e-4)

print_number_params(llm_model)

def train_baseline(num_epochs=50, batch_size=128):
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
    for epoch in range(num_epochs):
        # Mélange les données pour chaque epoch
        random.shuffle(train_data)
        total_loss = 0.0
        num_batches = len(train_data) // batch_size  # ~31 pour 4000/128
        for batch_idx in range(num_batches):
            batch_samples = train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            src_list = []
            tgt_list = []
            for input_str, target_str in batch_samples:
                src = torch.tensor([char2idx[BOS]] + [char2idx[c] for c in input_str], device=DEVICE)
                tgt = torch.tensor([char2idx[BOS]] + [char2idx[c] for c in target_str] + [char2idx[EOS]], device=DEVICE)
                src_list.append(src)
                tgt_list.append(tgt)
            # Pad les séquences pour batch
            src_padded = pad_sequence(src_list, batch_first=True, padding_value=char2idx[PAD])
            tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=char2idx[PAD])
            tgt_input = tgt_padded[:, :-1]
            tgt_output = tgt_padded[:, 1:].contiguous().view(-1)
            logits = llm_model(src_padded, tgt_input).contiguous().view(-1, VOCAB_SIZE)
            loss = criterion(logits, tgt_output)
            llm_opt.zero_grad()
            loss.backward()
            llm_opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Baseline Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        if avg_loss < 0.5:
            break

#train_baseline()

torch.save(llm_model.state_dict(), "baseline_llm.pth")

gflow_model = FlowNet().to(DEVICE)
gflow_opt = optim.Adam(gflow_model.parameters(), lr=3e-4)

print_number_params(gflow_model)


def train_adversarial(num_epochs=20, batch_size=64, mix_ratio=0.5):
    # mix_ratio: fraction generated vs. real data (cursor for efficiency test)
    for epoch in range(num_epochs):
        # Phase 1: GFlowNet maximizes divergence, preserves causal
        gflow_losses, divergences = [], []
        trajs, rewards = [], []
        for _ in range(batch_size):
            traj, reward = sample_trajectory(gflow_model, env)
            prefix = torch.tensor([traj[:-1]], device=DEVICE)
            token = traj[-1]
            prior_prob = 1.0 / len(traj)
            action_prob = random.uniform(0.5, 1.0)  # Dummy P(Action)
            info_prob = random.uniform(0.5, 1.0)  # Dummy P(Info|Action)
            div = compute_bayesian_divergence(llm_model, prefix, torch.tensor(token, device=DEVICE), prior_prob, action_prob, info_prob)
            reward += div  # Augment reward with div
            trajs.append(traj)
            rewards.append(reward)
            divergences.append(div)
        gflow_loss = tb_loss(gflow_model, trajs, rewards)
        gflow_opt.zero_grad()
        gflow_loss.backward()
        gflow_opt.step()
        gflow_losses.append(gflow_loss.item())

        # Phase 2: LLM minimizes on mixed batch (unsupervised for generated)
        llm_losses = []
        num_generated = int(batch_size * mix_ratio)
        num_real = batch_size - num_generated
        batch = []

        # Generated part (unsupervised: compute target if valid)
        for _ in range(num_generated):
            traj, _ = sample_trajectory(gflow_model, env)
            seq_str = ''.join(idx2char[t] for t in traj[1:-1])
            target_str = parse_and_compute_target(seq_str + '=')  # Add = for parse
            if target_str:
                src = torch.tensor([traj[:-1]], device=DEVICE)  # Prefix
                tgt = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in target_str] + [char2idx[EOS]]], device=DEVICE)
                batch.append((src, tgt))

        # Real data injection
        for input_str, target_str in random.sample(train_data, num_real):
            src = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in input_str]], device=DEVICE)
            tgt = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in target_str] + [char2idx[EOS]]], device=DEVICE)
            batch.append((src, tgt))

        for src, tgt in batch:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].view(-1)
            logits = llm_model(src, tgt_input).view(-1, VOCAB_SIZE)
            ce_loss = criterion(logits, tgt_output)
            # Divergence penalty (with new term)
            token = tgt_output[-1].item()  # Last
            prior_prob = 1.0 / src.size(1)
            action_prob = 1.0  # Placeholder
            info_prob = 1.0
            div_penalty = compute_bayesian_divergence(llm_model, src, torch.tensor(token, device=DEVICE), prior_prob, action_prob, info_prob)
            loss = ce_loss + 0.1 * div_penalty  # Reduce weight
            llm_opt.zero_grad()
            loss.backward()
            llm_opt.step()
            llm_losses.append(loss.item())

        print(f"Epoch {epoch+1} (mix={mix_ratio}): GFlow Loss={np.mean(gflow_losses):.4f}, Avg Div={np.mean(divergences):.4f}, LLM Loss={np.mean(llm_losses):.4f}")

train_adversarial(mix_ratio=0.5)  # Test with 50% mix; adjust cursor
torch.save(llm_model.state_dict(), "adversarial_llm.pth")

# Testing
def test_robustness(model, test_data, noise_level=0.1):
    model.eval()
    correct = 0
    for idx, (input_str, target_str) in enumerate(test_data[:100]):
        noisy_input = ''.join(random.choice('0123456789+= ') if random.random() < noise_level else c for c in input_str)
        generated = generate(model, noisy_input)
        expected = target_str
        if generated.strip() == expected:
            correct += 1
        if idx < 5:
            print(f"Example {idx}: prompt={noisy_input}, generated={generated}, expected={expected}")
    accuracy = correct / 100
    print(f"Robustness accuracy on noisy test: {accuracy:.2%}")
    return accuracy

print("Baseline robustness:")
baseline_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
baseline_model.load_state_dict(torch.load("baseline_llm.pth"))
baseline_acc = test_robustness(baseline_model, test_data)

print("Adversarial robustness:")
llm_model.load_state_dict(torch.load("adversarial_llm.pth"))
adversarial_acc = test_robustness(llm_model, test_data)

if adversarial_acc > baseline_acc:
    print("Success: Adversarial is more robust!")
else:
    print("Iterate: Try different mix_ratio.")
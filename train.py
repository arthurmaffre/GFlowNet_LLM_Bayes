import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from constants import DEVICE, char2idx, BOS, EOS, PAD
from models import Seq2SeqTransformer, FlowNet
from dataset import load_or_generate_dataset
from env import AddSeqEnv
from utils import sample_trajectory, tb_loss, compute_bayesian_divergence, parse_and_compute_target

full_data = load_or_generate_dataset()
train_data = full_data[:4000]
test_data = full_data[4000:]

def train_baseline(llm_model: Seq2SeqTransformer, num_epochs=50, batch_size=128):
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
    optimizer = optim.Adam(llm_model.parameters(), lr=5e-4)
    for epoch in range(num_epochs):
        llm_model.train()
        total_loss = 0.0
        num_batches = len(train_data) // batch_size
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            srcs, tgts = [], []
            for input_str, target_str in batch:
                src = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in input_str], device=DEVICE).unsqueeze(0)
                tgt = torch.tensor([char2idx[BOS]] + [char2idx.get(c, char2idx[PAD]) for c in target_str] + [char2idx[EOS]], device=DEVICE).unsqueeze(0)
                srcs.append(src)
                tgts.append(tgt)
            srcs = torch.cat(srcs, dim=0)
            tgts = torch.cat(tgts, dim=0)
            tgt_input = tgts[:, :-1]
            tgt_output = tgts[:, 1:].reshape(-1)
            logits = llm_model(srcs, tgt_input).reshape(-1, llm_model.fc.out_features)
            loss = criterion(logits, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Baseline Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        if avg_loss < 0.5:
            break
    torch.save(llm_model.state_dict(), "baseline_llm.pth")

def train_adversarial(llm_model: Seq2SeqTransformer, gflow_model: FlowNet, env, num_epochs=20, batch_size=64, mix_ratio=0.5):
    llm_optimizer = optim.Adam(llm_model.parameters(), lr=5e-4)
    gflow_optimizer = optim.Adam(gflow_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
    for epoch in range(num_epochs):
        # Phase 1: GFlowNet maximizes divergence
        gflow_model.train()
        gflow_losses, divergences = [], []
        trajs, rewards = [], []
        for _ in range(batch_size):
            traj, reward = sample_trajectory(gflow_model, env)
            prefix = torch.tensor([traj[:-1]], device=DEVICE)
            token = traj[-1]
            prior_prob = 1.0 / len(traj)  # Adaptive: could use token freq from corpus
            div = compute_bayesian_divergence(llm_model, prefix, token, prior_prob, gflow_model)
            reward += div
            trajs.append(traj)
            rewards.append(reward)
            divergences.append(div)
        gflow_loss = tb_loss(gflow_model, trajs, rewards)
        gflow_optimizer.zero_grad()
        gflow_loss.backward()
        gflow_optimizer.step()
        gflow_losses.append(gflow_loss.item())

        # Phase 2: LLM minimizes on mixed batch
        llm_model.train()
        llm_losses = []
        num_generated = int(batch_size * mix_ratio)
        num_real = batch_size - num_generated
        batch = []

        # Generated (unsupervised)
        for _ in range(num_generated):
            traj, _ = sample_trajectory(gflow_model, env)
            seq_str = ''.join(idx2char[t] for t in traj[1:-1])
            target_str = parse_and_compute_target(seq_str)
            if target_str:  # Skip invalid
                src = torch.tensor([traj[:-1]], device=DEVICE)
                tgt = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in target_str] + [char2idx[EOS]]], device=DEVICE)
                batch.append((src, tgt))

        # Real
        real_samples = random.sample(train_data, min(num_real, len(train_data)))
        for input_str, target_str in real_samples:
            src = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in input_str]], device=DEVICE)
            tgt = torch.tensor([[char2idx[BOS]] + [char2idx[c] for c in target_str] + [char2idx[EOS]]], device=DEVICE)
            batch.append((src, tgt))

        for src, tgt in batch:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].view(-1)
            logits = llm_model(src, tgt_input).view(-1, llm_model.fc.out_features)
            ce_loss = criterion(logits, tgt_output)
            token = tgt_output[0].item()  # First non-BOS for simplicity
            prior_prob = 1.0 / src.size(1)
            div_penalty = compute_bayesian_divergence(llm_model, src, token, prior_prob, gflow_model)
            loss = ce_loss + 0.1 * div_penalty
            llm_optimizer.zero_grad()
            loss.backward()
            llm_optimizer.step()
            llm_losses.append(loss.item())

        print(f"Epoch {epoch+1} (mix={mix_ratio}): GFlow Loss={np.mean(gflow_losses):.4f}, Avg Div={np.mean(divergences):.4f}, LLM Loss={np.mean(llm_losses):.4f}")
    torch.save(llm_model.state_dict(), "adversarial_llm.pth")
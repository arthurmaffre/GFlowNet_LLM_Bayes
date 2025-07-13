from constants import VOCAB_SIZE, char2idx, PAD, DEVICE
from models import Seq2SeqTransformer, FlowNet
from env import AddSeqEnv
from train import train_baseline, train_adversarial
from test import test_robustness
from dataset import load_or_generate_dataset

import torch

print(f"Vocab size: {VOCAB_SIZE}, Device: {DEVICE}")

data = load_or_generate_dataset()
print(f"Generated/Loaded {len(data)} samples. Example: {data[0]}")

env = AddSeqEnv()

# Baseline
baseline_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
train_baseline(baseline_model)
baseline_acc = test_robustness(baseline_model, data[4000:])

# Adversarial
llm_model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
llm_model.load_state_dict(torch.load("baseline_llm.pth"))  # Start from baseline
gflow_model = FlowNet().to(DEVICE)
train_adversarial(llm_model, gflow_model, env, mix_ratio=0.5)
adversarial_acc = test_robustness(llm_model, data[4000:])

if adversarial_acc > baseline_acc:
    print("Success: Adversarial is more robust!")
else:
    print("Iterate: Try different mix_ratio.")
"""Minimal GFlowNet-like sampling for the addition transformer."""
from typing import List

import torch

from model import Seq2SeqTransformer, load_dataset, char2idx, idx2char, VOCAB_SIZE, PAD, BOS, EOS, train


def sample_action(model: Seq2SeqTransformer, max_len: int = 9, device: str = "cpu") -> List[int]:
    """Sample an input sequence token-by-token using the model."""
    ids = []
    x = torch.tensor([[char2idx[BOS]]], device=device)
    for _ in range(max_len):
        logits = model.fc(model.emb(x).sum(1))
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids.append(next_id.item())
        x = torch.cat([x, next_id], dim=1)
        if idx2char[next_id.item()] == EOS:
            break
    return ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("addition_dataset.pkl")
    model = Seq2SeqTransformer(VOCAB_SIZE).to(device)
    train(model, ds, device, epochs=1)

    for _ in range(5):
        seq = sample_action(model, device=device)
        print("Sampled:", ''.join(idx2char[i] for i in seq))


if __name__ == "__main__":
    main()

import pickle
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

SPECIAL = ["<pad>", "<bos>", "<eos>"]
BASE_CHARS = sorted(set("0123456789+= "))
VOCAB = SPECIAL + BASE_CHARS
PAD, BOS, EOS = SPECIAL
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(VOCAB)
INPUT_LEN = 9
OUTPUT_LEN = 2  # maximum number of digits in a+b (for 0-49)
TGT_LEN = OUTPUT_LEN + 2


def load_dataset(path: str) -> Dataset:
    with open(path, "rb") as f:
        df = pickle.load(f)
    return AdditionDataset(df)


class AdditionDataset(Dataset):
    def __init__(self, dataframe):
        self.x, self.y = [], []
        for _, row in dataframe.iterrows():
            inp = row["input"][:INPUT_LEN].ljust(INPUT_LEN)
            out = row["output"][:OUTPUT_LEN].ljust(OUTPUT_LEN)
            src_ids = [char2idx[c] for c in inp]
            tgt_ids = [char2idx[BOS]] + [char2idx[c] for c in out] + [char2idx[EOS]]
            tgt_ids += [char2idx[PAD]] * (TGT_LEN - len(tgt_ids))
            self.x.append(torch.tensor(src_ids, dtype=torch.long))
            self.y.append(torch.tensor(tgt_ids, dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=char2idx[PAD])
        self.pos = nn.Embedding(512, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dropout=0.1)
        dec_layer = nn.TransformerDecoderLayer(emb_dim, n_heads, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def add_pos(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.emb(x) + self.pos(positions)

    def forward(self, src, tgt):
        src_emb = self.add_pos(src)
        tgt_emb = self.add_pos(tgt)
        memory = self.encoder(src_emb.permute(1, 0, 2))
        out = self.decoder(tgt_emb.permute(1, 0, 2), memory)
        return self.fc(out.permute(1, 0, 2))


def train(model: Seq2SeqTransformer, dataset: Dataset, device: str = "cpu", epochs: int = 5) -> None:
    train_len = int(0.8 * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lengths)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])

    for epoch in range(1, epochs + 1):
        model.train(); total_tr = 0
        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_tr += loss.item()
        train_loss = total_tr / len(train_dl)

        model.eval(); total_te = 0
        with torch.no_grad():
            for src, tgt in test_dl:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt[:, :-1])
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
                total_te += loss.item()
        test_loss = total_te / len(test_dl)
        print(f"Epoch {epoch:02d} – Train {train_loss:.4f} | Test {test_loss:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("addition_dataset.pkl")
    model = Seq2SeqTransformer(VOCAB_SIZE).to(device)
    train(model, ds, device, epochs=10)

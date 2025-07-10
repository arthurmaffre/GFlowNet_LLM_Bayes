import torch
import torch.nn as nn
from constants import VOCAB_SIZE, DEVICE, char2idx, idx2char, PAD, MAX_LEN

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, n_heads: int = 2, n_layers: int = 2, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(512, emb_dim)
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
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

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
    














if __name__ == "__main__":
    import torch.optim as optim
    from torch.nn import CrossEntropyLoss
    from torch.distributions import Categorical
    
    print(f"Testing models on device: {DEVICE}")
    
    # Tiny dataset for Seq2SeqTransformer
    tiny_data = [("1 + 1 =", "2"), ("2 + 3 =", "5")]
    model = Seq2SeqTransformer(VOCAB_SIZE, pad_idx=char2idx[PAD]).to(DEVICE)
    criterion = CrossEntropyLoss(ignore_index=char2idx[PAD])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    for epoch in range(5):  # Short training
        total_loss = 0.0
        for input_str, target_str in tiny_data:
            src = torch.tensor([[char2idx.get(c, char2idx[PAD]) for c in input_str]], device=DEVICE)
            tgt = torch.tensor([[char2idx.get(c, char2idx[PAD]) for c in target_str]], device=DEVICE)
            tgt_input = tgt[:, :-1] if tgt.size(1) > 1 else tgt
            tgt_output = tgt[:, 1:].view(-1) if tgt.size(1) > 1 else tgt.view(-1)
            logits = model(src, tgt_input).view(-1, VOCAB_SIZE)
            loss = criterion(logits, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(tiny_data)
        if epoch == 0:
            initial_loss = avg_loss
        final_loss = avg_loss
    print(f"Seq2SeqTransformer: Initial loss = {initial_loss:.4f}, Final loss = {final_loss:.4f}")
    if final_loss < initial_loss:
        print("✓ Loss decreased as expected.")
    else:
        print("✗ Loss did not decrease.")
    
    # Simple forward pass for FlowNet
    flow_model = FlowNet().to(DEVICE)
    dummy_prefix = torch.tensor([[char2idx['1'], char2idx['+']]], device=DEVICE)
    log_probs = flow_model(dummy_prefix)
    action = Categorical(logits=log_probs).sample().item()
    generated_char = idx2char.get(action, "<unknown>")
    print(f"FlowNet sample: Prefix '1 +', next char '{generated_char}'")
    if log_probs.shape == (1, VOCAB_SIZE):
        print("✓ Output shape correct.")
    else:
        print("✗ Output shape incorrect.")
    
    print("Model tests completed.")
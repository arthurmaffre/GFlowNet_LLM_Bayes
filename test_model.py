import torch
from model import (
    Seq2SeqTransformer,
    load_dataset,
    VOCAB_SIZE,
    evaluate,
    char2idx,
    idx2char,
    PAD,
    BOS,
    EOS,
    TGT_LEN,
    train,
)


def decode_ids(ids):
    chars = []
    for i in ids:
        ch = idx2char[int(i)]
        if ch == EOS:
            break
        if ch not in (BOS, PAD):
            chars.append(ch)
    return "".join(chars)


def generate(model, src, device="cpu"):
    model.eval()
    src = src.unsqueeze(0).to(device)
    tgt = torch.tensor([[char2idx[BOS]]], device=device)
    with torch.no_grad():
        for _ in range(TGT_LEN):
            out = model(src, tgt)
            next_id = out[0, -1].argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_id], dim=1)
            if next_id.item() == char2idx[EOS]:
                break
    return tgt.squeeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("addition_dataset.pkl")
    model = Seq2SeqTransformer(VOCAB_SIZE).to(device)
    train(model, ds, device, epochs=10)
    acc = evaluate(model, ds, device)
    print(f"Token accuracy on dataset: {acc:.4f}")
    for i in range(5):
        src, _ = ds[i]
        pred = generate(model, src.to(device), device)
        inp_str = decode_ids(src)
        out_str = decode_ids(pred)
        print(f"{inp_str} -> {out_str}")


if __name__ == "__main__":
    main()

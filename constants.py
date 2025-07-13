import torch

SPECIAL = ["<pad>", "<bos>", "<eos>"]
BASE_CHARS = list("0123456789+= ")
VOCAB = SPECIAL + BASE_CHARS
PAD, BOS, EOS = SPECIAL
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(VOCAB)
MAX_LEN = 12
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
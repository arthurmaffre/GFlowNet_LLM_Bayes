import torch
import random
#from constants import DEVICE, char2idx, PAD
from models import Seq2SeqTransformer
from utils import generate
from dataset import load_or_generate_dataset

full_data = load_or_generate_dataset()
test_data = full_data[4000:]

def test_robustness(model, test_data, noise_level=0.1, num_samples=100):
    model.eval()
    correct = 0
    for idx, (input_str, target_str) in enumerate(test_data[:num_samples]):
        noisy_input = ''.join(random.choice('0123456789+= ') if random.random() < noise_level else c for c in input_str)
        generated = generate(model, noisy_input)
        if generated.strip() == target_str:
            correct += 1
        if idx < 5:
            print(f"Example {idx}: prompt={noisy_input}, generated={generated}, expected={target_str}")
    accuracy = correct / num_samples
    print(f"Robustness accuracy on noisy test: {accuracy:.2%}")
    return accuracy
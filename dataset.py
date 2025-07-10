import random
import pickle
import os
from typing import List, Tuple

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

def load_or_generate_dataset(file_path: str = "addition_dataset.pkl") -> List[Tuple[str, str]]:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        data = generate_addition_dataset()
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        return data

if __name__ == "__main__":
    data = load_or_generate_dataset()
    print(f"Dataset size: {len(data)} samples")
    
    # Verify integrity with random samples
    samples_to_check = 5
    random_samples = random.sample(data, samples_to_check)
    all_correct = True
    for i, (input_str, target_str) in enumerate(random_samples, 1):
        try:
            parts = input_str.split('+')
            a = int(parts[0].strip())
            b = int(parts[1].split('=')[0].strip())
            expected = str(a + b)
            correct = expected == target_str
            status = "✓" if correct else "✗"
            print(f"Sample {i}: {input_str} {target_str} {status}")
            if not correct:
                all_correct = False
        except ValueError:
            print(f"Sample {i}: Invalid format {input_str}")
            all_correct = False
    
    if all_correct:
        print("All checked samples are correct.")
    else:
        print("Some samples have errors.")
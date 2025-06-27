import random
import pickle
from typing import List, Tuple


def generate_addition_dataset(
    num_samples: int = 1000, max_val: int = 49
) -> List[Tuple[str, str]]:
    """Generate a list of ``(input, output)`` pairs for ``a + b``."""
    data: List[Tuple[str, str]] = []
    for _ in range(num_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        result = a + b
        data.append((f"{a} + {b}", str(result)))
    return data


def save_dataset(data: List[Tuple[str, str]], path: str = "addition_dataset.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    data = generate_addition_dataset()
    save_dataset(data)
    print(f"Dataset saved to addition_dataset.pkl with {len(data)} samples")

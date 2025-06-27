import random
import pandas as pd
import pickle


def generate_addition_dataset(num_samples: int = 1000, max_val: int = 49) -> pd.DataFrame:
    """Generate a simple (a + b, result) dataset."""
    data = []
    for _ in range(num_samples):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        result = a + b
        data.append((f"{a} + {b}", str(result)))
    return pd.DataFrame(data, columns=["input", "output"])


def save_dataset(df: pd.DataFrame, path: str = "addition_dataset.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    df = generate_addition_dataset()
    save_dataset(df)
    print(f"Dataset saved to addition_dataset.pkl with {len(df)} samples")

import random
import pickle
from typing import List, Tuple, Optional, Dict, Any


"""
=========================================================
Dataset Generator — Addition Tasks
=========================================================

Auteur : Arthur Maffre
Description :
--------------
Ce script génère un dataset complet d’additions (a + b = c)
pour entraîner ou tester des modèles de séquence (Seq2Seq, LLM, etc.).
Il produit deux fichiers `.pkl` :
  - `addition_dataset_train.pkl`
  - `addition_dataset_eval.pkl`

Le split est déterministe :
  → l’ensemble d’évaluation (eval) contient uniquement les additions
     où `a` et `b` sont tous deux compris entre [40, 49].
  → tout le reste va dans l’ensemble d’entraînement (train).

Chaque fichier contient :
  - "data": liste de tuples (entrée, sortie), ex. ("43+66=", "109")
  - "metadata": informations globales sur le dataset (vocabulaire, tailles, seed, etc.)

Structure :
------------
1️⃣ Génération exhaustive du dataset
2️⃣ Split logique train/eval
3️⃣ Construction des métadonnées
4️⃣ Sauvegarde des datasets
5️⃣ Vérification d’intégrité complète
6️⃣ Script principal avec résumé et aperçu

=========================================================
"""

# =========================================================
# 1️⃣ Génération du dataset
# =========================================================
def generate_addition_dataset(max_val: int = 99, seed: Optional[int] = None) -> List[Tuple[str, str]]:
    """Génère toutes les combinaisons possibles de a + b = c dans [0, max_val].
    - Si `seed` est spécifiée → shuffle reproductible.
    - Sinon → shuffle aléatoire.
    """
    data: List[Tuple[str, str]] = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            result = a + b
            input_str = f"{a}+{b}="
            target_str = str(result)
            data.append((input_str, target_str))

    if seed is not None:
        rnd = random.Random(seed)
        rnd.shuffle(data)
    else:
        random.shuffle(data)

    return data


# =========================================================
# 2️⃣ Split train / eval
# =========================================================
def split_train_eval(data: List[Tuple[str, str]], eval_range: Tuple[int, int] = (40, 49)) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Sépare le dataset :
       - eval : si a ∈ [eval_range] ou b ∈ [eval_range]
       - train : sinon
    """
    train_data, eval_data = [], []
    min_r, max_r = eval_range

    for x, y in data:
        a_str, b_str = x.replace("=", "").split("+")
        a, b = int(a_str), int(b_str)
        if min_r <= a <= max_r and min_r <= b <= max_r:
            eval_data.append((x, y))
        else:
            train_data.append((x, y))

    return train_data, eval_data


# =========================================================
# 3️⃣ Métadonnées du dataset
# =========================================================
def build_metadata(data: List[Tuple[str, str]], max_val: int, seed: Optional[int]) -> Dict[str, Any]:
    """Construit les métadonnées utiles pour le dataset."""
    inputs, outputs = zip(*data)
    max_len_input = max(len(s) for s in inputs)
    max_len_output = max(len(s) for s in outputs)

    # Construction du vocabulaire à partir de tous les caractères uniques
    all_chars = sorted(list(set("".join(inputs + outputs))))
    vocabulary = {ch: i for i, ch in enumerate(all_chars)}

    metadata = {
        "MAX_LEN_INPUT": max_len_input,
        "MAX_LEN_OUTPUT": max_len_output,
        "VOCABULARY": vocabulary,
        "VOCAB_SIZE": len(vocabulary),
        "MIN_NUM": 0,
        "MAX_NUM": max_val,
        "SEED": seed,
    }

    return metadata


# =========================================================
# 4️⃣ Sauvegarde
# =========================================================
def save_dataset(filename: str, data: List[Tuple[str, str]], metadata: Dict[str, Any]):
    """Sauvegarde les données et les métadonnées dans un fichier pickle."""
    with open(filename, "wb") as f:
        pickle.dump({"data": data, "metadata": metadata}, f)
    print(f"✅ Saved {filename} ({len(data)} samples)")


# =========================================================
# 5️⃣ Vérification d'intégrité
# =========================================================
def verify_dataset_integrity(train_path: str, eval_path: str, eval_range: Tuple[int, int], max_val: int):
    """Vérifie la cohérence entre les métadonnées et les données."""
    print("\n🧪 Vérification d'intégrité du dataset...")

    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(eval_path, "rb") as f:
        eval_ = pickle.load(f)

    train_data, train_meta = train["data"], train["metadata"]
    eval_data, eval_meta = eval_["data"], eval_["metadata"]

    # 1. Vérifie le total attendu
    total_expected = (max_val + 1) ** 2
    total_actual = len(train_data) + len(eval_data)
    assert total_actual == total_expected, f"❌ Total mismatch: expected {total_expected}, got {total_actual}"

    # 2. Vérifie la cohérence des longueurs
    def check_lengths(data, meta):
        in_max = max(len(x) for x, _ in data)
        out_max = max(len(y) for _, y in data)
        assert in_max == meta["MAX_LEN_INPUT"], f"❌ MAX_LEN_INPUT mismatch ({in_max} vs {meta['MAX_LEN_INPUT']})"
        assert out_max <= meta["MAX_LEN_OUTPUT"], f"❌ MAX_LEN_OUTPUT mismatch ({out_max} vs {meta['MAX_LEN_OUTPUT']})"

    check_lengths(train_data, train_meta)
    check_lengths(eval_data, eval_meta)

    # 3. Vérifie la cohérence du split
    min_r, max_r = eval_range
    for x, _ in train_data:
        a_str, b_str = x.replace("=", "").split("+")
        a, b = int(a_str), int(b_str)
        assert not (min_r <= a <= max_r and min_r <= b <= max_r), f"❌ Train data contains eval range: {x}"

    for x, _ in eval_data:
        a_str, b_str = x.replace("=", "").split("+")
        a, b = int(a_str), int(b_str)
        assert (min_r <= a <= max_r and min_r <= b <= max_r), f"❌ Eval data outside range: {x}"

    # 4. Vérifie la cohérence du vocabulaire
    assert train_meta["VOCABULARY"] == eval_meta["VOCABULARY"], "❌ Vocabulary mismatch between train and eval"

    print("✅ Vérification réussie — dataset intègre et cohérent.")


# =========================================================
# 6️⃣ Script principal
# =========================================================
if __name__ == "__main__":
    MAX_VAL = 99
    SEED = 42
    EVAL_RANGE = (40, 49)

    print("🔧 Génération du dataset...")
    data = generate_addition_dataset(max_val=MAX_VAL, seed=SEED)
    train_data, eval_data = split_train_eval(data, eval_range=EVAL_RANGE)
    metadata = build_metadata(data, max_val=MAX_VAL, seed=SEED)

    save_dataset("addition_dataset_train.pkl", train_data, metadata)
    save_dataset("addition_dataset_eval.pkl", eval_data, metadata)

    print(f"\n📊 Résumé :")
    print(f"Train size: {len(train_data)}")
    print(f"Eval size:  {len(eval_data)}")
    print(f"Max input len: {metadata['MAX_LEN_INPUT']}")
    print(f"Max output len: {metadata['MAX_LEN_OUTPUT']}")
    print(f"Vocab size: {metadata['VOCAB_SIZE']}")

    print("\n🔍 Exemples aléatoires (train):")
    for x, y in random.sample(train_data, 3):
        print(f"  {x}{y}")

    print("\n🔍 Exemples aléatoires (eval):")
    for x, y in random.sample(eval_data, 3):
        print(f"  {x}{y}")

    # Vérification d'intégrité
    verify_dataset_integrity(
        "addition_dataset_train.pkl",
        "addition_dataset_eval.pkl",
        eval_range=EVAL_RANGE,
        max_val=MAX_VAL
    )
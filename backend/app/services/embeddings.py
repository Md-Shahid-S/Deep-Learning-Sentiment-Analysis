import numpy as np
from pathlib import Path

def load_glove(glove_path: str, vocab: dict, embed_dim: int = 100) -> np.ndarray:
    """
    Build embedding matrix from GloVe file.
    vocab: {word: index} dict built during training
    Returns: np.ndarray of shape (vocab_size, embed_dim)
    """
    embed_matrix = np.zeros((len(vocab), embed_dim))
    found = 0

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                embed_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"GloVe coverage: {found}/{len(vocab)} words ({found/len(vocab)*100:.1f}%)")
    return embed_matrix
from typing import List

import numpy as np


def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """حساب التشابه بين بصمتين"""
    v1 = np.array(emb1)
    v2 = np.array(emb2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))
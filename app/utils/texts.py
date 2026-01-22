import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
  # Retorna una matriu (n_texts, dim)
  vectors = model.encode(texts, normalize_embeddings=True)
  return np.array(vectors, dtype=np.float32)

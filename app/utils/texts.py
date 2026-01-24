import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


VECTOR_SIZE = os.getenv("VECTOR_SIZE", 384)


def embed_text(model: SentenceTransformer, text: str) -> List[float]:
  vec = model.encode([text], normalize_embeddings=True)[0]
  vec = np.array(vec, dtype=np.float32)

  if vec.shape[0] != VECTOR_SIZE:
    raise RuntimeError(
      f"Embedding dim mismatch: got {vec.shape[0]}, expected {VECTOR_SIZE}. "
      "Check your model or collection config."
    )

  return vec.tolist()

import numpy as np
from typing import List


def softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
  # Converteix la llista de "scores" amb una llista de probabilitats (la suma de totes és 100%)
  if not scores:
    return []

  t = max(temperature, 1e-6)
  x = np.array(scores, dtype=np.float64) / t

  # estabilitat numèrica: restem el màxim per evitar overflows
  x = x - np.max(x)

  exps = np.exp(x)
  probs = exps / np.sum(exps)
  return probs.tolist()

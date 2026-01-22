import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.models.recipe import Recipe
from app.stores.feedback_store import FeedbackStore
from app.utils.texts import embed_texts


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
  # Amb embeddings normalitzats, el cosinus és simplement el dot product
  # Calcular el producte escalar de 2 vectors
  return float(np.dot(a, b))


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


def build_recipe_centroids(
  recipes: List[Recipe],
  base_vecs: np.ndarray,
  model: SentenceTransformer,
  feedback_store: FeedbackStore,
  max_prompts_per_recipe: int = 50
) -> np.ndarray:
  # Inicialment centroid = base_vec
  centroids = np.array(base_vecs, dtype=np.float32)

  # Agrupem prompts "positius" per recipe_id final
  records = feedback_store.read_all()
  prompts_by_recipe_id = {}

  for rec in records:
    status = rec.get("status")
    final_recipe_id = rec.get("final_recipe_id")
    prompt = rec.get("prompt", "")

    if not prompt or not final_recipe_id:
      continue

    # Considerem accepted i corrected com a positius per la recepta final
    if status in ("accepted", "corrected"):
      prompts_by_recipe_id.setdefault(final_recipe_id, []).append(prompt)

  # Recalculem centroid per cada recepta
  recipe_id_to_index = {r.id: i for i, r in enumerate(recipes)}

  for recipe_id, prompts in prompts_by_recipe_id.items():
    if recipe_id not in recipe_id_to_index:
      continue

    i = recipe_id_to_index[recipe_id]
    # Limitem prompts per no “sobre-entrenar” en aquesta demo
    prompts = prompts[-max_prompts_per_recipe:]

    prompt_vecs = embed_texts(model, prompts)  # (n, dim)
    combined = np.vstack([centroids[i][None, :], prompt_vecs])  # (n+1, dim)

    centroid = np.mean(combined, axis=0)

    # IMPORTANT: normalitzem perquè el dot torni a ser cosine
    norm = np.linalg.norm(centroid)
    if norm > 0:
      centroid = centroid / norm

    centroids[i] = centroid.astype(np.float32)

  return centroids

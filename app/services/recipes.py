import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.models.recipe import Recipe
from app.core.mongodb_connector import MongoDBConnector
from app.utils.texts import embed_texts
from app.utils.embeddings import cosine_sim, softmax

def load_recipes_from_mongo() -> List[Recipe]:
  mongo_connector = MongoDBConnector()
  recipes_collection = mongo_connector.db["recipes"]

  docs = list(recipes_collection.find({}))
  recipes: List[Recipe] = []

  for d in docs:
    recipes.append(
      Recipe(
        id=str(d["_id"]),
        name=d.get("name", "").strip(),
        description=d.get("description", "").strip()
      )
    )

  if not recipes:
    raise RuntimeError("No s'han trobat receptes a Mongo. Inserta almenys 1 recepta a la collection 'recipes'.")

  return recipes


def rank_recipes(
  prompt: str,
  recipes: List[Recipe],
  recipe_vecs: np.ndarray,
  model: SentenceTransformer,
  top_k: int = 5
) -> List[tuple]:
  prompt_vec = embed_texts(model, [prompt])[0]  # shape (384,)
  scored = []

  for i, r in enumerate(recipes):
    score = cosine_sim(prompt_vec, recipe_vecs[i])
    scored.append((r, score))

  scored.sort(key=lambda x: x[1], reverse=True)
  return scored[:max(1, min(top_k, len(scored)))]

def predict_recipes(
  prompt: str,
  recipes: List[Recipe],
  recipe_vecs: np.ndarray,
  model: SentenceTransformer,
  top_k: int = 5,
  temperature: float = 0.2
) -> List[tuple]:
  ranked = rank_recipes(
    prompt=prompt,
    recipes=recipes,
    recipe_vecs=recipe_vecs,
    model=model,
    top_k=top_k
  )

  scores = [score for _, score in ranked]
  probs = softmax(scores, temperature=temperature)

  # Retornem tuples: (Recipe, score, prob)
  enriched = []
  for i, (r, score) in enumerate(ranked):
    prob = probs[i] if i < len(probs) else 0.0
    enriched.append((r, score, prob))

  return enriched

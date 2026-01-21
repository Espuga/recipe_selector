import os
from dataclasses import dataclass
from typing import List
import numpy as np
import json

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "recipes")
RECIPES_COLLECTION = os.getenv("RECIPES_COLLECTION", "recipes")
EMBEDDING_MODEL_NAME = os.getenv(
  "EMBEDDING_MODEL_NAME",
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback.jsonl")



@dataclass(frozen=True)
class Recipe:
  id: str
  name: str
  description: str
  def base_text(self) -> str:
    return f"{self.name}. {self.description}".strip()

class FeedbackStore:
  def __init__(self, path: str):
    self.path = path

  def append(self, record: dict) -> None:
    with open(self.path, "a", encoding="utf-8") as f:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

  def read_all(self) -> List[dict]:
    if not os.path.exists(self.path):
      return []

    records = []
    with open(self.path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          records.append(json.loads(line))
        except json.JSONDecodeError:
          continue

    return records

def load_recipes_from_mongo() -> List[Recipe]:
  client = MongoClient(MONGO_URI)
  db = client[DB_NAME]
  col = db[RECIPES_COLLECTION]

  docs = list(col.find({}))
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


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
  # Retorna una matriu (n_texts, dim)
  vectors = model.encode(texts, normalize_embeddings=True)
  return np.array(vectors, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
  # Amb embeddings normalitzats, el cosinus és simplement el dot product
  # Calcular el producte escalar de 2 vectors
  return float(np.dot(a, b))


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


def main() -> None:
  recipes = load_recipes_from_mongo()
  print(f"Loaded {len(recipes)} recipe(s)")

  model = SentenceTransformer(EMBEDDING_MODEL_NAME)
  feedback_store = FeedbackStore(FEEDBACK_PATH)

  # Base embeddings (estables)
  recipe_texts = [r.base_text() for r in recipes]
  base_vecs = embed_texts(model, recipe_texts)

  # Centroids (aprèn del feedback)
  centroid_vecs = build_recipe_centroids(
    recipes=recipes,
    base_vecs=base_vecs,
    model=model,
    feedback_store=feedback_store
  )

  print(f"Embedding dim: {base_vecs.shape[1]}")
  print("\nEscriu un prompt en català. Per sortir: 'exit'\n")

  while True:
    prompt = input("Prompt> ").strip()
    if not prompt:
      continue
    if prompt.lower() in ("exit", "quit"):
      break

    predicted = predict_recipes(
      prompt=prompt,
      recipes=recipes,
      recipe_vecs=centroid_vecs,  # IMPORTANT: centroids!
      model=model,
      top_k=5,
      temperature=0.2
    )

    best_r, best_score, best_prob = predicted[0]
    print(f"\nBest: {best_r.name} (score={best_score:.3f}, confidence={best_prob:.2f})")

    print("\nTop matches:")
    for r, score, prob in predicted:
      print(f"  - {r.name:25s} score={score:.3f} prob={prob:.2f}")

    print("\nFeedback:")
    print("  [Enter] = correcte (acceptat per defecte)")
    print("  n       = no és correcte (rebutjat)")
    print("  c       = corregir (tria una recepta del top)")

    fb = input("Feedback> ").strip().lower()

    predicted_id = best_r.id

    if fb == "":
      feedback_store.append({
        "prompt": prompt,
        "predicted_recipe_id": predicted_id,
        "status": "accepted",
        "final_recipe_id": predicted_id
      })
      print("✅ Guardat com ACCEPTAT.\n")

    elif fb == "n":
      feedback_store.append({
        "prompt": prompt,
        "predicted_recipe_id": predicted_id,
        "status": "rejected",
        "final_recipe_id": predicted_id
      })
      print("❌ Guardat com REBUTJAT.\n")

    elif fb == "c":
      print("\nTria la recepta correcta:")
      for i, (r, score, prob) in enumerate(predicted, start=1):
        print(f"  {i}) {r.name} (prob={prob:.2f})")

      choice = input("Número> ").strip()
      try:
        idx = int(choice) - 1
        idx = max(0, min(idx, len(predicted) - 1))
      except ValueError:
        idx = 0

      final_r = predicted[idx][0]
      feedback_store.append({
        "prompt": prompt,
        "predicted_recipe_id": predicted_id,
        "status": "corrected",
        "final_recipe_id": final_r.id
      })
      print(f"🛠️  Corregit. Correcte = {final_r.name}\n")

    else:
      print("Feedback no reconegut. No s'ha guardat.\n")

    # Recalcula centroids després del feedback (aprèn en calent)
    centroid_vecs = build_recipe_centroids(
      recipes=recipes,
      base_vecs=base_vecs,
      model=model,
      feedback_store=feedback_store
    )


if __name__ == "__main__":
  main()
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "recipes")
RECIPES_COLLECTION = os.getenv("RECIPES_COLLECTION", "recipes")

# Guardem feedback local per aquesta prova (més endavant ho podem portar a Mongo)
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback.jsonl")

# Model multilingüe (CAT/ENG/etc.)
# És una opció molt comuna per embeddings multilingües.
EMBEDDING_MODEL_NAME = os.getenv(
  "EMBEDDING_MODEL_NAME",
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@dataclass(frozen=True)
class Recipe:
  id: str
  name: str
  description: str

  def base_text(self) -> str:
  # Text canònic: si després hi afegeixes tags, exemples, etc. els concatenes aquí
    return f"{self.name}. {self.description}".strip()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(v)
  if norm == 0:
    return v
  return v / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
  a = _l2_normalize(a)
  b = _l2_normalize(b)
  return float(np.dot(a, b))


def softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
  if not xs:
    return []
  t = max(temperature, 1e-6)
  arr = np.array(xs, dtype=np.float64) / t
  arr = arr - np.max(arr)
  exps = np.exp(arr)
  probs = exps / np.sum(exps)
  return probs.tolist()


class FeedbackStore:
  def __init__(self, path: str):
    self.path = path

  def append(self, record: dict) -> None:
    with open(self.path, "a", encoding="utf-8") as f:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")

  def iter_records(self) -> List[dict]:
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


class RecipeSelector:
  def __init__(self, recipes: List[Recipe], model: SentenceTransformer, feedback_store: FeedbackStore):
    self.recipes = recipes
    self.model = model
    self.feedback_store = feedback_store

    # Centroid per recepta: comencem amb embedding del text base de la recepta.
    self.recipe_centroids: Dict[str, np.ndarray] = {}
    self._build_centroids_from_base()
    self._apply_feedback_learning()

  def _embed(self, text: str) -> np.ndarray:
    v = self.model.encode([text], normalize_embeddings=True)[0]
    return np.array(v, dtype=np.float32)

  def _build_centroids_from_base(self) -> None:
    for r in self.recipes:
      self.recipe_centroids[r.id] = self._embed(r.base_text())

  def _apply_feedback_learning(self) -> None:
    """
    “Aprendre” del feedback:
    - per cada recepta, agafem prompts acceptats/correctes
    - recalcularem un centroid com la mitjana (base + prompts)
    """
    records = self.feedback_store.iter_records()
    prompts_by_recipe: Dict[str, List[str]] = {}

    for rec in records:
      # Tipus:
      # - accepted: l'usuari no diu res o confirma
      # - corrected: l'usuari tria una altra recepta
      status = rec.get("status")
      final_recipe_id = rec.get("final_recipe_id")
      prompt = rec.get("prompt", "")
      if not prompt or not final_recipe_id:
        continue

      if status in ("accepted", "corrected"):
        prompts_by_recipe.setdefault(final_recipe_id, []).append(prompt)

    for recipe_id, prompts in prompts_by_recipe.items():
      if recipe_id not in self.recipe_centroids:
        continue

      vectors = [self.recipe_centroids[recipe_id]]
      # Per evitar que un usuari “domini” el centroid amb mil prompts,
      # limitem a un màxim per demo (ajustable)
      for p in prompts[-50:]:
        vectors.append(self._embed(p))

      centroid = np.mean(np.stack(vectors, axis=0), axis=0)
      self.recipe_centroids[recipe_id] = _l2_normalize(centroid)

  def rank(self, prompt: str, top_k: int = 3) -> List[Tuple[Recipe, float]]:
    prompt_vec = self._embed(prompt)
    scored: List[Tuple[Recipe, float]] = []

    for r in self.recipes:
      c = self.recipe_centroids.get(r.id)
      if c is None:
        c = self._embed(r.base_text())
      score = cosine_sim(prompt_vec, c)
      scored.append((r, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, min(top_k, len(scored)))]

  def predict(self, prompt: str, top_k: int = 3) -> dict:
    ranked = self.rank(prompt, top_k=top_k)
    scores = [s for _, s in ranked]
    probs = softmax(scores, temperature=0.2)  # temperatura baixa = més “decidit”

    best_recipe, best_score = ranked[0]
    confidence = probs[0] if probs else 0.0

    return {
      "best": {
        "recipe_id": best_recipe.id,
        "name": best_recipe.name,
        "description": best_recipe.description,
        "score": best_score,
        "confidence": confidence
      },
      "alternatives": [
        {
          "recipe_id": r.id,
          "name": r.name,
          "score": s,
          "prob": probs[i] if i < len(probs) else None
        }
        for i, (r, s) in enumerate(ranked)
      ]
    }

  def record_feedback(self, prompt: str, predicted_recipe_id: str, status: str, final_recipe_id: str) -> None:
    record = {
      "prompt": prompt,
      "predicted_recipe_id": predicted_recipe_id,
      "status": status,
      "final_recipe_id": final_recipe_id
    }
    self.feedback_store.append(record)

    # Recalcular centroids en calent (demo). En producció ho faries batch o amb cache.
    self._apply_feedback_learning()


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


def main() -> None:
  recipes = load_recipes_from_mongo()
  print(f"Loaded {len(recipes)} recipe(s) from Mongo.")

  model = SentenceTransformer(EMBEDDING_MODEL_NAME)
  feedback_store = FeedbackStore(FEEDBACK_PATH)
  selector = RecipeSelector(recipes, model, feedback_store)

  print("\nEscriu un prompt en català. Exemple: 'vull enviar una notificació'")
  print("Per sortir: 'exit'\n")

  while True:
    prompt = input("Prompt> ").strip()
    if not prompt:
      continue
    if prompt.lower() in ("exit", "quit"):
      break

    pred = selector.predict(prompt, top_k=3)
    best = pred["best"]

    print("\nPredicció:")
    print(f"  -> {best['name']}  (confidence={best['confidence']:.2f}, score={best['score']:.3f})")

    if pred["alternatives"]:
      print("  Alternatives:")
      for alt in pred["alternatives"]:
        prob_str = f"{alt['prob']:.2f}" if alt["prob"] is not None else "n/a"
        print(f"    - {alt['name']} (prob={prob_str}, score={alt['score']:.3f})")

    print("\nFeedback:")
    print("  [Enter] = correcte (acceptat per defecte)")
    print("  n       = no és correcte (rebutjat)")
    print("  c       = corregir (tria una recepta del top)")

    fb = input("Feedback> ").strip().lower()

    predicted_id = best["recipe_id"]

    if fb == "":
      selector.record_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="accepted",
        final_recipe_id=predicted_id
      )
      print("✅ Guardat com ACCEPTAT.\n")
      continue

    if fb == "n":
      selector.record_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="rejected",
        final_recipe_id=predicted_id
      )
      print("❌ Guardat com REBUTJAT (sense correcció).\n")
      continue

    if fb == "c":
      alts = pred["alternatives"]
      if not alts:
        print("No hi ha alternatives disponibles.\n")
        continue

      print("\nTria una recepta:")
      for i, alt in enumerate(alts, start=1):
        print(f"  {i}) {alt['name']}")

      choice = input("Número> ").strip()
      try:
        idx = int(choice) - 1
        idx = max(0, min(idx, len(alts) - 1))
      except ValueError:
        idx = 0

      final_id = alts[idx]["recipe_id"]
      selector.record_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="corrected",
        final_recipe_id=final_id
      )
      print(f"🛠️  Corregit. Ara el sistema aprendrà que això correspon a: {alts[idx]['name']}\n")
      continue

    print("Feedback no reconegut. No s'ha guardat.\n")


if __name__ == "__main__":
  main()
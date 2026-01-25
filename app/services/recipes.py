from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client.http import models as qm
from bson import ObjectId
import numpy as np

from app.core.qdrant_connector import QdrantConnector
from app.core.mongodb_connector import MongoDBConnector
from app.utils.texts import embed_text
from app.utils.embeddings import  softmax
from app.utils.qdrant import build_filter_app_or_generic
from app.models.recipe import Recipe
from app.services.recipes_qdrant import update_recipe_vector_from_feedback


mongodb_connector = MongoDBConnector()
qdrant_connector = QdrantConnector()


def get_recipe_by_id(recipe_id: str) -> Recipe:
  recipes_collection = mongodb_connector.db["recipes"]
  recipe = recipes_collection.find_one({"_id": ObjectId(recipe_id)})
  recipe["id"] = str(recipe["_id"])
  del recipe["_id"]
  if "app_id" in recipe:
    recipe["app_id"] = str(recipe["app_id"])
  return Recipe(**recipe)


def search_recipes(
  model: SentenceTransformer,
  prompt: str,
  top_k: int,
  app_id: Optional[str],
  temperature: float = 0.2
) -> List[Dict[str, Any]]:

  query_vec = embed_text(model, prompt)

  query_filter = None
  if app_id:
    query_filter = build_filter_app_or_generic(app_id)

  results = qdrant_connector.client.query_points(
    collection_name=qdrant_connector.COLLECTION_NAME,
    query=query_vec,
    limit=top_k,
    query_filter=query_filter,
    with_payload=True
  ).points
  print("\n=== Results ===")
  for r in results:
    print(f"- {r.payload['name']}")
  print("===============")

  scores = [r.score for r in results]
  probs = softmax(scores, temperature=temperature)

  enriched = []
  for i, r in enumerate(results):
    prob = probs[i] if i < len(probs) else 0.0
    payload = r.payload or {}
    enriched.append((payload.get("recipe_id"), r.score, prob))

  return enriched


def add_recipe_feedback(
  model: SentenceTransformer,
  predicted_recipe_id: str,
  prompt: str,
  status: str,
  final_recipe_id: str = None,
):
  if final_recipe_id is None:
    final_recipe_id = predicted_recipe_id

  use_recipes_feedback_collection = mongodb_connector.db["use_recipes_feedback"]
  use_recipes_feedback_collection.insert_one({
    "predicted_recipe_id": ObjectId(predicted_recipe_id),
    "prompt": prompt,
    "status": status,
    "final_recipe_id": ObjectId(final_recipe_id)
  })

  if status in ("accepted", "corrected"):
    update_recipe_vector_from_feedback(
      model=model,  # millor si passes model ja creat
      recipe_id=final_recipe_id,
      max_prompts=50,
      base_weight=10.0
    )

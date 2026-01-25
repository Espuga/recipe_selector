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

def add_recipe_to_qdrant(
  model: SentenceTransformer,
  recipe_id: str,
  name: str,
  description: str,
  app_id: Optional[str]
) -> str:

  point_id = str(uuid.uuid4())  # Qdrant point id (UUID)
  base_text = f"{name}. {description}".strip()
  vector = embed_text(model, base_text)

  payload: Dict[str, Any] = {
    "recipe_id": recipe_id,  # mandatory
    "name": name,
    "description": description
  }
  if app_id:
    payload["app_id"] = app_id

  qdrant_connector.client.upsert(
    collection_name=qdrant_connector.COLLECTION_NAME,
    points=[
      qm.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
      )
    ]
  )

  return point_id


def find_point_ids_by_recipe_id(recipe_id: str, limit: int = 100) -> List[str]:

  flt = qm.Filter(
    must=[
      qm.FieldCondition(
        key="recipe_id",
        match=qm.MatchValue(value=recipe_id)
      )
    ]
  )

  point_ids: List[str] = []
  offset = None

  # Scroll pages until we collect all matches (up to limit per page)
  while True:
    points, next_offset = qdrant_connector.client.scroll(
      collection_name=qdrant_connector.COLLECTION_NAME,
      scroll_filter=flt,
      limit=min(100, limit),
      offset=offset,
      with_payload=False,
      with_vectors=False
    )

    for p in points:
      point_ids.append(str(p.id))

    if not next_offset or len(point_ids) >= limit:
      break

    offset = next_offset

  return point_ids[:limit]


def delete_recipe_by_recipe_id_qdrant(recipe_id: str) -> int:

  point_ids = find_point_ids_by_recipe_id(recipe_id, limit=1000)
  if not point_ids:
    return 0

  qdrant_connector.client.delete(
    collection_name=qdrant_connector.COLLECTION_NAME,
    points_selector=qm.PointIdsList(points=point_ids)
  )
  return len(point_ids)


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

def update_recipe_vector_from_feedback(
  model: SentenceTransformer,
  recipe_id: str,
  max_prompts: int = 50,
  base_weight: float = 10.0
) -> None:
  print("Updating recipe vector...")
  recipe = get_recipe_by_id(recipe_id)
  print(f"Recipe name: {recipe.name}")

  # 1) Base vector (name + description)
  base_text = f"{recipe.name}. {recipe.description}".strip()
  print(f"Base text: {base_text}")
  base_vec = np.array(embed_text(model, base_text), dtype=np.float32)

  # 2) Agafa els últims N prompts positius de Mongo
  feedback_col = mongodb_connector.db["use_recipes_feedback"]

  cursor = feedback_col.find(
    {
      "final_recipe_id": ObjectId(recipe_id),
      "status": {"$in": ["accepted", "corrected"]}
    },
    {"prompt": 1}
  ).sort([("_id", -1)]).limit(max_prompts)

  prompts = [doc.get("prompt", "").strip() for doc in cursor if doc.get("prompt")]
  prompts = [p for p in prompts if p]
  print("Prompts")
  for p in prompts:
    print(f"- {p}")

  # 3) Embeddings dels prompts
  if prompts:
    prompt_vecs = np.array([embed_text(model, p) for p in prompts], dtype=np.float32)
    sum_prompt = np.sum(prompt_vecs, axis=0)
    centroid = (base_vec * base_weight + sum_prompt) / (base_weight + float(len(prompts)))
  else:
    centroid = base_vec

  # 4) Normalitza (perquè cosine/dot funcioni bé)
  norm = float(np.linalg.norm(centroid))
  if norm > 0:
    centroid = centroid / norm

  centroid_list = centroid.astype(np.float32).tolist()

  # 5) Troba point(s) a Qdrant per recipe_id
  point_ids = find_point_ids_by_recipe_id(recipe_id, limit=1000)

  # 6) Si no existeix, crea un point
  if not point_ids:
    add_recipe_to_qdrant(
      model=model,
      recipe_id=recipe_id,
      name=recipe.name,
      description=recipe.description,
      app_id=recipe.app_id
    )
    point_ids = find_point_ids_by_recipe_id(recipe_id, limit=1000)

  if not point_ids:
    # Si encara no hi és, alguna cosa rara ha passat
    raise RuntimeError(f"No s'ha pogut crear/trobar cap point a Qdrant per recipe_id={recipe_id}")

  # 7) Actualitza el vector del primer point (upsert amb el mateix UUID)
  keep_point_id = point_ids[0]

  payload: Dict[str, Any] = {
    "recipe_id": recipe_id,
    "name": recipe.name,
    "description": recipe.description
  }
  # app_id pot ser None en receptes genèriques
  if getattr(recipe, "app_id", None):
    payload["app_id"] = recipe.app_id

  qdrant_connector.client.upsert(
    collection_name=qdrant_connector.COLLECTION_NAME,
    points=[
      qm.PointStruct(
        id=keep_point_id,
        vector=centroid_list,
        payload=payload
      )
    ]
  )

  # 8) Si hi havia duplicats, elimina els altres
  if len(point_ids) > 1:
    qdrant_connector.client.delete(
      collection_name=qdrant_connector.COLLECTION_NAME,
      points_selector=qm.PointIdsList(points=point_ids[1:])
    )
  print("done")


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

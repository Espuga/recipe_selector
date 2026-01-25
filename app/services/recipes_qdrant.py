from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client.http import models as qm
from bson import ObjectId
import numpy as np

from app.core.qdrant_connector import QdrantConnector
from app.core.mongodb_connector import MongoDBConnector
from app.utils.texts import embed_text
from app.models.recipe import Recipe


mongodb_connector = MongoDBConnector()
qdrant_connector = QdrantConnector()


recipes_collection = mongodb_connector.db["recipes"]


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


def update_recipe_vector_from_feedback(
  model: SentenceTransformer,
  recipe: Recipe,
  max_prompts: int = 50,
  base_weight: float = 10.0
) -> None:
  print("Updating recipe vector...")

  # 1) Base vector (name + description)
  base_text = f"{recipe.name}. {recipe.description}".strip()
  base_vec = np.array(embed_text(model, base_text), dtype=np.float32)

  # 2) Agafa els últims N prompts positius de Mongo
  feedback_col = mongodb_connector.db["use_recipes_feedback"]

  cursor = feedback_col.find(
    {
      "final_recipe_id": ObjectId(recipe.id),
      "status": {"$in": ["accepted", "corrected"]}
    },
    {"prompt": 1}
  ).sort([("_id", -1)]).limit(max_prompts)

  prompts = [doc.get("prompt", "").strip() for doc in cursor if doc.get("prompt")]
  prompts = [p for p in prompts if p]

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
  point_ids = find_point_ids_by_recipe_id(recipe.id, limit=1000)

  # 6) Si no existeix, crea un point
  if not point_ids:
    add_recipe_to_qdrant(
      model=model,
      recipe_id=recipe.id,
      name=recipe.name,
      description=recipe.description,
      app_id=recipe.app_id
    )
    point_ids = find_point_ids_by_recipe_id(recipe.id, limit=1000)

  if not point_ids:
    # Si encara no hi és, alguna cosa rara ha passat
    raise RuntimeError(f"No s'ha pogut crear/trobar cap point a Qdrant per recipe_id={recipe.id}")

  # 7) Actualitza el vector del primer point (upsert amb el mateix UUID)
  keep_point_id = point_ids[0]

  payload: Dict[str, Any] = {
    "recipe_id": recipe.id,
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


from typing import Optional
from bson import ObjectId
from sentence_transformers import SentenceTransformer

def sync_recipe_by_id(
  model: SentenceTransformer,
  recipe_id: str,
  *,
  max_prompts: int = 50,
  base_weight: float = 10.0
) -> Recipe:
  # Accepta recipe_id com str d'ObjectId o com "id" intern en string
  try:
    oid = ObjectId(recipe_id)
  except Exception as e:
    raise ValueError(f"recipe_id no és un ObjectId vàlid: {recipe_id}") from e

  doc = recipes_collection.find_one({"_id": oid})
  if not doc:
    raise LookupError(f"No existeix cap recepta a Mongo amb _id={recipe_id}")

  recipe = Recipe.from_mongo(doc)

  # Aquesta funció (la teva) ja:
  # - calcula vector amb name/description + prompts
  # - crea point a Qdrant si no existeix
  # - fa upsert del vector/payload
  # - elimina duplicats
  update_recipe_vector_from_feedback(
    model=model,
    recipe=recipe,
    max_prompts=max_prompts,
    base_weight=base_weight
  )

  return recipe



def sync_recipes(
  model: SentenceTransformer,
  *,
  max_prompts: int = 50,
  base_weight: float = 10.0
) -> None:
  cursor = recipes_collection.find({}, {"_id": 1})

  ok = 0
  failed = 0

  for doc in cursor:
    rid = str(doc["_id"])
    try:
      sync_recipe_by_id(
        model=model,
        recipe_id=rid,
        max_prompts=max_prompts,
        base_weight=base_weight
      )
      ok += 1
    except Exception as e:
      failed += 1
      print(f"[sync_recipes] ERROR recipe_id={rid}: {e}")

  print(f"[sync_recipes] done. ok={ok}, failed={failed}")


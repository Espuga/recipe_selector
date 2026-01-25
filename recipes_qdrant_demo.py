import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")

EMBEDDING_MODEL_NAME = os.getenv(
  "EMBEDDING_MODEL_NAME",
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))


def prompt_input(label: str, required: bool = False, default: str = "") -> str:
  while True:
    suffix = f" [{default}]" if default else ""
    val = input(f"{label}{suffix}: ").strip()
    if not val and default:
      val = default
    if required and not val:
      print("❌ Aquest camp és obligatori.")
      continue
    return val


def get_client() -> QdrantClient:
  return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_model() -> SentenceTransformer:
  return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed(model: SentenceTransformer, text: str) -> List[float]:
  vec = model.encode([text], normalize_embeddings=True)[0]
  vec = np.array(vec, dtype=np.float32)

  if vec.shape[0] != VECTOR_SIZE:
    raise RuntimeError(
      f"Embedding dim mismatch: got {vec.shape[0]}, expected {VECTOR_SIZE}. "
      "Check your model or collection config."
    )

  return vec.tolist()


def ensure_collection_and_payload_indexes(client: QdrantClient) -> None:
  existing = {c.name for c in client.get_collections().collections}
  if COLLECTION_NAME not in existing:
    client.create_collection(
      collection_name=COLLECTION_NAME,
      vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE)
    )

  # Payload indexes (keyword) for exact filtering:
  # - recipe_id: mandatory (Mongo ObjectId string)
  # - app_id: optional
  # If they already exist, Qdrant will ignore or return an error depending on version.
  # We'll be tolerant and just try.
  for field in ("recipe_id", "app_id"):
    try:
      client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=qm.PayloadSchemaType.KEYWORD
      )
    except Exception:
      pass


def build_filter_app_or_generic(app_id: str) -> qm.Filter:
  # (app_id == X) OR (app_id is empty/absent)
  return qm.Filter(
    should=[
      qm.FieldCondition(
        key="app_id",
        match=qm.MatchValue(value=app_id)
      ),
      qm.IsEmptyCondition(is_empty=qm.PayloadField(key="app_id"))
    ]
  )


def add_recipe(
  client: QdrantClient,
  model: SentenceTransformer,
  recipe_id: str,
  name: str,
  description: str,
  app_id: Optional[str]
) -> str:
  ensure_collection_and_payload_indexes(client)

  point_id = str(uuid.uuid4())  # Qdrant point id (UUID)
  base_text = f"{name}. {description}".strip()
  vector = embed(model, base_text)

  payload: Dict[str, Any] = {
    "recipe_id": recipe_id,  # mandatory
    "name": name,
    "description": description
  }
  if app_id:
    payload["app_id"] = app_id

  client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
      qm.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload
      )
    ]
  )

  return point_id


def find_point_ids_by_recipe_id(client: QdrantClient, recipe_id: str, limit: int = 100) -> List[str]:
  ensure_collection_and_payload_indexes(client)

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
    points, next_offset = client.scroll(
      collection_name=COLLECTION_NAME,
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


def delete_recipe_by_recipe_id(client: QdrantClient, recipe_id: str) -> int:
  ensure_collection_and_payload_indexes(client)

  point_ids = find_point_ids_by_recipe_id(client, recipe_id, limit=1000)
  if not point_ids:
    return 0

  client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=qm.PointIdsList(points=point_ids)
  )
  return len(point_ids)


def search_recipes(
  client: QdrantClient,
  model: SentenceTransformer,
  prompt: str,
  top_k: int,
  app_id: Optional[str]
) -> List[Dict[str, Any]]:
  ensure_collection_and_payload_indexes(client)

  query_vec = embed(model, prompt)

  query_filter = None
  if app_id:
    query_filter = build_filter_app_or_generic(app_id)

  results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vec,
    limit=top_k,
    query_filter=query_filter,
    with_payload=True
  ).points

  out: List[Dict[str, Any]] = []
  for r in results:
    payload = r.payload or {}
    out.append({
      "qdrant_id": str(r.id),
      "score": float(r.score),
      "recipe_id": payload.get("recipe_id"),
      "name": payload.get("name"),
      "description": payload.get("description"),
      "app_id": payload.get("app_id")  # may be None
    })

  return out


def main() -> None:
  client = get_client()
  model = get_model()

  print("Qdrant Recipes Demo (interactive)")
  print(f"- Qdrant: http://{QDRANT_HOST}:{QDRANT_PORT}")
  print(f"- Collection: {COLLECTION_NAME}")
  print(f"- Embedding model: {EMBEDDING_MODEL_NAME}")
  print("")

  ensure_collection_and_payload_indexes(client)

  while True:
    print("\nOpcions:")
    print("  1) Afegir recepta (genera UUID a Qdrant)")
    print("  2) Eliminar recepta (per recipe_id payload)")
    print("  3) Buscar receptes (search)")
    print("  4) Sortir")
    choice = input("Tria una opció (1-4): ").strip()

    if choice == "4":
      print("Adéu!")
      break

    if choice == "1":
      print("\n-- Afegir recepta --")
      recipe_id = prompt_input("recipe_id (Mongo ObjectId com string)", required=True)
      name = prompt_input("name", required=True)
      description = prompt_input("description", required=True)
      app_id = prompt_input("app_id (opcional, buit = genèrica)", required=False).strip() or None

      point_id = add_recipe(
        client=client,
        model=model,
        recipe_id=recipe_id,
        name=name,
        description=description,
        app_id=app_id
      )
      print(f"✅ OK: afegida. qdrant_id={point_id}")
      continue

    if choice == "2":
      print("\n-- Eliminar recepta --")
      recipe_id = prompt_input("recipe_id (Mongo ObjectId com string)", required=True)

      deleted = delete_recipe_by_recipe_id(client, recipe_id)
      if deleted == 0:
        print("ℹ️  No s'ha trobat cap point amb aquest recipe_id.")
      else:
        print(f"✅ OK: eliminats {deleted} point(s) amb recipe_id={recipe_id}.")
      continue

    if choice == "3":
      print("\n-- Buscar receptes --")
      prompt = prompt_input("prompt (CAT)", required=True)
      top_k_str = prompt_input("top_k", required=False, default="5")
      try:
        top_k = int(top_k_str)
        if top_k < 1:
          top_k = 5
      except ValueError:
        top_k = 5

      app_id = prompt_input(
        "app_id (opcional; si el poses fa: app_id==X OR sense app_id)",
        required=False
      ).strip() or None

      results = search_recipes(
        client=client,
        model=model,
        prompt=prompt,
        top_k=top_k,
        app_id=app_id
      )

      if not results:
        print("No hi ha resultats.")
        continue

      print("\nResultats:")
      for i, r in enumerate(results, start=1):
        print(
          f"{i}) score={r['score']:.4f} recipe_id={r.get('recipe_id')} "
          f"name={r.get('name')} app_id={r.get('app_id')} qdrant_id={r['qdrant_id']}"
        )
      continue

    print("Opció no vàlida. Tria 1, 2, 3 o 4.")


if __name__ == "__main__":
  main()
import os
from sentence_transformers import SentenceTransformer
from app.services.recipes import load_recipes_from_mongo
from app.stores.feedback_store import FeedbackStore
from app.utils.texts import embed_texts
from app.utils.embeddings import build_recipe_centroids
from app.services.recipes import predict_recipes

def main():
  recipes = load_recipes_from_mongo()
  print(f"Loaded {len(recipes)} recipe(s)")

  EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  )
  FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback.jsonl")

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

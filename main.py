import os
from sentence_transformers import SentenceTransformer
from app.services.recipes import search_recipes, get_recipe_by_id, add_recipe_feedback


def main():

  EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  )

  model = SentenceTransformer(EMBEDDING_MODEL_NAME)
  print("\nEscriu un prompt en català. Per sortir: 'exit'\n")

  while True:
    prompt = input("Prompt> ").strip()
    if not prompt:
      continue
    if prompt.lower() in ("exit", "quit"):
      break

    predicted = search_recipes(
      model=model,
      prompt=prompt,
      top_k=3,
      app_id="697486305742fbea6a5615fc",
    )

    best_r_id, best_score, best_prob = predicted[0]
    recipe = get_recipe_by_id(best_r_id)
    print(f"\nBest: {recipe.name} (score={best_score:.3f}, confidence={best_prob:.2f})")

    print("\nTop matches:")
    for r_id, score, prob in predicted:
      recipe = get_recipe_by_id(r_id)
      print(f"  - {recipe.name:25s} score={score:.3f} prob={prob:.2f}")

    print("\nFeedback:")
    print("  [Enter] = correcte (acceptat per defecte)")
    print("  n       = no és correcte (rebutjat)")
    print("  c       = corregir (tria una recepta del top)")

    fb = input("Feedback> ").strip().lower()

    predicted_id = best_r_id
    if fb == "":
      add_recipe_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="accepted"
      )
      print("✅ Guardat com ACCEPTAT.\n")

    elif fb == "n":
      add_recipe_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="rejected"
      )
      print("❌ Guardat com REBUTJAT.\n")

    elif fb == "c":
      print("\nTria la recepta correcta:")
      for i, (r_id, score, prob) in enumerate(predicted, start=1):
        r = get_recipe_by_id(r_id)
        print(f"  {i}) {r.name} (prob={prob:.2f})")

      choice = input("Número> ").strip()
      try:
        idx = int(choice) - 1
        idx = max(0, min(idx, len(predicted) - 1))
      except ValueError:
        idx = 0

      final_r_id = predicted[idx][0]
      add_recipe_feedback(
        prompt=prompt,
        predicted_recipe_id=predicted_id,
        status="corrected",
        final_recipe_id=final_r_id
      )
      print(f"🛠️  Corregit. Correcte = {final_r_id}\n")

    else:
      print("Feedback no reconegut. No s'ha guardat.\n")


if __name__ == "__main__":
  main()

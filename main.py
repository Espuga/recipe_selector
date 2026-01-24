import os
from sentence_transformers import SentenceTransformer
from app.services.recipes import search_recipes, get_recipe_by_id


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

    print("\nTop matches:")
    for r_id, score, prob in predicted:
      recipe = get_recipe_by_id(r_id)
      print(f"  - {recipe.name:25s} score={score:.3f} prob={prob:.2f}")


if __name__ == "__main__":
  main()

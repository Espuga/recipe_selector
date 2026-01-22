# AGENTS.md

## Project overview
This repository is a recipe selection demo using embeddings. It loads recipes from MongoDB, generates embeddings with `sentence-transformers`, classifies a Catalan prompt, and learns from feedback to recalculate recipe centroids.

## Folder and file structure

### Repository root
- `main.py`: CLI entry point. Loads recipes, computes embeddings, makes predictions, and collects feedback to recalculate centroids.
- `README.md`: project overview and usage.
- `requirements.txt`: Python dependencies (`pymongo`, `sentence-transformers`, `numpy`).
- `.env.example`: expected environment variables.
- `documentation/recipe_chooser.drawio`: flow diagram (Draw.io file).

### `app/`
Main project package.

#### `app/models/`
- `recipe.py`: `Recipe` dataclass with `id`, `name`, `description`, and `base_text()`.

#### `app/services/`
- `recipes.py`: domain logic to load recipes from MongoDB, rank, and predict recipes with probabilities.

#### `app/core/`
- `mongodb_connector.py`: simple MongoDB connector (`MongoClient`) and database selection.

#### `app/stores/`
- `feedback_store.py`: feedback persistence in JSONL format (append and read).

#### `app/utils/`
- `texts.py`: helpers to generate normalized embeddings.
- `embeddings.py`: cosine similarity (with normalized embeddings), temperature softmax, and centroid computation from feedback.

## Main data flow
1. `main.py` loads recipes from MongoDB (`recipes` collection).
2. Generates base embeddings with `SentenceTransformer`.
3. Computes centroids per recipe using accepted/corrected feedback.
4. Classifies the prompt with similarity and shows top results.
5. Saves feedback to `feedback.jsonl` (configurable) and recalculates centroids.

## Environment variables and configuration
- `EMBEDDING_MODEL_NAME`: `sentence-transformers` model.
- `FEEDBACK_PATH`: path to the feedback JSONL file.

See `.env.example` for defaults.

## Dependencies
Install with:
```
pip install -r requirements.txt
```

## Runtime requirements
- MongoDB accessible (default `mongodb://localhost:27017`).
- `recipes` collection inside the `recipes` database with documents that include `name` and `description`.

## Run
```
python main.py
```

## Notes for contributors
- This is a CLI demo; no automated tests are included.
- Feedback is appended to JSONL and used to learn centroids.

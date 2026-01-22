# AGENTS.md

## Visió general del projecte
Aquest repositori és una demo de selecció de receptes amb embeddings. Carrega receptes des de MongoDB, genera embeddings amb `sentence-transformers`, classifica un prompt en català i aprèn del feedback per recalcular centroides de receptes.

## Estructura de carpetes i fitxers

### Arrel del repositori
- `main.py`: punt d'entrada CLI. Carrega receptes, calcula embeddings, fa prediccions i recull feedback per recalcular centroides.
- `README.md`: descripció mínima del projecte.
- `requirements.txt`: dependències Python (`pymongo`, `sentence-transformers`, `numpy`).
- `.env.example`: variables d'entorn esperades.
- `documentation/recipe_chooser.drawio`: diagrama del flux (fitxer Draw.io).

### `app/`
Paquet principal del projecte.

#### `app/models/`
- `recipe.py`: dataclass `Recipe` amb `id`, `name`, `description` i `base_text()`.

#### `app/services/`
- `recipes.py`: lògica de domini per carregar receptes de MongoDB, rankejar i predir receptes amb probabilitats.

#### `app/core/`
- `mongodb_connector.py`: connector senzill a MongoDB (`MongoClient`) i selecció de base de dades.

#### `app/stores/`
- `feedback_store.py`: persistència de feedback en format JSONL (append i lectura).

#### `app/utils/`
- `texts.py`: helpers per generar embeddings normalitzats.
- `embeddings.py`: cosine similarity (amb embeddings normalitzats), softmax amb temperatura i càlcul de centroides a partir de feedback.

## Flux principal de dades
1. `main.py` carrega receptes de MongoDB (`recipes` collection).
2. Genera embeddings base amb `SentenceTransformer`.
3. Calcula centroides per recepta amb feedback acceptat/corregit.
4. Classifica el prompt amb similarity i mostra top resultats.
5. Desa feedback en `feedback.jsonl` (configurable) i recalcula centroides.

## Variables d'entorn i configuració
- `EMBEDDING_MODEL_NAME`: model de `sentence-transformers`.
- `FEEDBACK_PATH`: ruta del fitxer JSONL de feedback.

Consulta `.env.example` per valors per defecte.

## Dependències
Instal·la amb:
```
pip install -r requirements.txt
```

## Requisits d'execució
- MongoDB accessible (per defecte a `mongodb://localhost:27017`).
- Col·lecció `recipes` dins la base de dades `recipes` amb documents que tinguin `name` i `description`.

## Execució
```
python main.py
```

## Notes per contribuir
- El projecte és una demo CLI; no hi ha tests automatitzats inclosos.
- El feedback s'acumula en JSONL i s'utilitza per aprendre centroides.

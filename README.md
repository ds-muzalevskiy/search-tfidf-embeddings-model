# search-tfidf-embeddings-model

# Document Search Service

FastAPI service with TF-IDF and embedding search.

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Adding document

```
curl -X POST http://localhost:8000/documents -H "Content-Type: application/json" -d '{"id":"doc1", "title":"AI", "content":"AI and ML are rising now"}
```

## Adding document
```
curl "http://localhost:8000/search?query=AI&method=tfidf"
```


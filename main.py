from fastapi import FastAPI, HTTPException, Query
from typing import List
from index import DocumentIndex
from model import Document, SearchResponse

app = FastAPI()
index = DocumentIndex()

@app.post("/documents")
def add_documents(doc: Document):
    try:
        index.add_documents(doc)
        return {"message": "Document added successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search", response_model=List[SearchResponse])
def search(query: str, method:str = Query("tfidf", title="Search method", description="Search method to use"), k: int = Query(10, title="Number of results", description="Number of results to return")):
    if method not in ['tfidf', 'embedding']:
        raise HTTPException(status_code=400, detail="Invalid search method")
    try:
        return index.search(query, method, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.delete("/documents")
def reset_index():
    index.reset()
    return {"message": "Index reset successfully"}
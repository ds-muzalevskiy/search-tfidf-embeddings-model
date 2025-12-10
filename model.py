from pydantic import BaseModel

class Document(BaseModel):
    title: str
    id: str
    content: str

class SearchResponse(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
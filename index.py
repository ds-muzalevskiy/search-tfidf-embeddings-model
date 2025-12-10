import numpy as np
from typing import List, Dict
from model import Document, SearchResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class DocumentIndex:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix: None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_matrix = None

    def add_documents(self, doc: Document):
        if doc.id in self.documents:
            raise ValueError(f"Document with this id already exists")
        self.documents[doc.id] = doc
        self.update_indices()

    def update_indices(self):
        contents = [d.content for d in self.documents.values()]
        if contents:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
            self.embedding_matrix = np.vstack(self.embedding_model.encode(contents, convert_to_numpy=True))

    def search(self, query: str, method:str, k: int) -> List[SearchResponse]:
        docs = list(self.documents.values())
        if not docs:
            return []

        if method == 'tfidf':
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        else:
            query_vec = self.embedding_model.encode([query], convert_to_numpy=True)
            scores = cosine_similarity(query_vec, self.embedding_matrix).flatten()

        top_indices = scores.argsort()[::-1][:k]
        results = []

        for i in top_indices:
            d = docs[i]
            results.append(SearchResponse(id=d.id,
                                          title=d.title,
                                          snippet=d.content[:100],
                                          score=float(scores[i])))
        return results

    def reset(self):
        self.documents = {}
        self.tfidf_matrix = None
        self.embedding_matrix = None


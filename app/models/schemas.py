from typing import List, Optional

from pydantic import BaseModel


class Base64Request(BaseModel):
    image_base64: str


class CompareRequest(BaseModel):
    emb1: List[float]
    emb2: List[float]
    threshold: float = 0.65


class SearchRequest(BaseModel):
    query_embedding: List[float]
    database_embeddings: List[List[float]]
    threshold: float = 0.65
    top_k: int = 50
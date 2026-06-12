from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import SearchRequest
from app.services.similarity import cosine_similarity

router = APIRouter(tags=["Search"])


@router.post("/search")
async def search_faces(payload: SearchRequest):
    """
    البحث عن أقرب وجوه مشابهة في قاعدة بيانات من البصمات
    - query_embedding: بصمة الوجه المطلوب البحث عنه
    - database_embeddings: قائمة بالبصمات المخزنة لديك
    - threshold: عتبة التشابه (0.65 - 0.7 جيد)
    - top_k: عدد النتائج القصوى
    """
    if not payload.database_embeddings:
        return {"matches": []}
    similarities = []
    for i, db_emb in enumerate(payload.database_embeddings):
        sim = cosine_similarity(payload.query_embedding, db_emb)
        similarities.append((i, sim))
    # ترتيب تنازلي
    similarities.sort(key=lambda x: x[1], reverse=True)
    results = []
    for i, sim in similarities[:payload.top_k]:
        if sim >= payload.threshold:
            results.append({"index": i, "similarity": sim})
    return {"matches": results, "count": len(results)}
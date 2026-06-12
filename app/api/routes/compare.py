from fastapi import APIRouter, File, HTTPException, UploadFile
from typing_extensions import Annotated

from app.core.config import settings
from app.models.schemas import CompareRequest
from app.services.face_analyzer import get_embedding
from app.services.similarity import cosine_similarity
from app.utils.image import read_image

router = APIRouter(tags=["Compare"])


@router.post("/compare")
async def compare_faces(payload: CompareRequest):
    """مقارنة بصمتين (ترسل كـ JSON)"""
    similarity = cosine_similarity(payload.emb1, payload.emb2)
    return {
        "similarity": similarity,
        "is_match": similarity >= payload.threshold,
        "threshold": payload.threshold
    }


@router.post("/compare-files")
async def compare_faces_files(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = settings.DEFAULT_THRESHOLD,
):
    """مقارنة وجهين عن طريق رفع صورتين"""
    img1 = read_image(file1)
    img2 = read_image(file2)
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    if emb1["count"] == 0 or emb2["count"] == 0:
        raise HTTPException(status_code=404, detail="No face in one or both images")
    similarity = cosine_similarity(emb1["faces"][0]["embedding"], emb2["faces"][0]["embedding"])
    return {
        "similarity": similarity,
        "is_match": similarity >= threshold,
        "threshold": threshold
    }
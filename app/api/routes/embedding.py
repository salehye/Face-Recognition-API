from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import Base64Request
from app.services.face_analyzer import get_embedding
from app.utils.image import base64_to_image, read_image

router = APIRouter(tags=["Embedding"])


@router.post("/embedding")
async def get_face_embedding(file: UploadFile = File(...)):
    """رفع صورة واستخراج بصمة الوجه الأول (أقوى وجه)"""
    img = read_image(file)
    result = get_embedding(img)
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No face detected in image")
    # نرجع أول وجه (يمكنك تعديل لترجع القائمة كلها)
    first_face = result["faces"][0]
    return {
        "success": True,
        "embedding": first_face["embedding"],
        "bbox": first_face["bbox"],
        "confidence": first_face["confidence"],
        "embedding_size": len(first_face["embedding"])
    }


@router.post("/embedding-base64")
async def get_face_embedding_base64(payload: Base64Request):
    """نفس الشي ولكن باستقبال base64 (للتطبيقات التي لا ترفع ملفات مباشرة)"""
    img = base64_to_image(payload.image_base64)
    result = get_embedding(img)
    if result["count"] == 0:
        raise HTTPException(status_code=404, detail="No face detected")
    first_face = result["faces"][0]
    return {
        "success": True,
        "embedding": first_face["embedding"],
        "bbox": first_face["bbox"],
        "confidence": first_face["confidence"]
    }
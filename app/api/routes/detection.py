from fastapi import APIRouter, File, UploadFile

from app.services.face_analyzer import face_app
from app.utils.image import get_face_coordinates, read_image

router = APIRouter(prefix="/detect", tags=["Detection"])


@router.post("")
async def detect_faces(file: UploadFile = File(...)):
    """كشف جميع الوجوه في الصورة مع إحداثياتها (بدون بصمات)"""
    img = read_image(file)
    faces = face_app.get(img)
    h, w = img.shape[:2]
    result = get_face_coordinates(faces, w, h)
    return {"faces": result, "count": len(result)}


@router.post("-embeddings")
async def detect_faces_with_embeddings(file: UploadFile = File(...)):
    """كشف جميع الوجوه مع إحداثياتها وبصماتها"""
    import numpy as np

    img = read_image(file)
    faces = face_app.get(img)
    # ترتيب الوجوه تنازلياً حسب المساحة ليكون الوجه الأكبر والأقرب في المقدمة دائماً
    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    h, w = img.shape[:2]
    results = []
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int).tolist()
        # L2 normalization of embedding to make it unit norm (norm = 1.0)
        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        results.append({
            "index": i,
            "embedding": embedding.tolist(),
            "bbox": {
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1],
                "normalized": {
                    "x": bbox[0] / w, "y": bbox[1] / h,
                    "width": (bbox[2] - bbox[0]) / w, "height": (bbox[3] - bbox[1]) / h
                }
            },
            "confidence": float(face.det_score)
        })
    return {"faces": results, "count": len(results)}
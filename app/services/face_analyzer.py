import numpy as np

from insightface.app import FaceAnalysis

from app.core.config import settings

# تحميل نموذج InsightFace (أدق نموذج متاح)
# ctx_id=0 للـ GPU، -1 للـ CPU
face_app = FaceAnalysis(name=settings.MODEL_NAME, providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(settings.DET_SIZE, settings.DET_SIZE))


def get_embedding(img: np.ndarray) -> dict:
    """استخراج بصمات جميع الوجوه في الصورة"""
    faces = face_app.get(img)
    # ترتيب الوجوه تنازلياً حسب المساحة ليكون الوجه الأكبر والأقرب في المقدمة دائماً
    faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    results = []
    h, w = img.shape[:2]
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        # L2 normalization of embedding to make it unit norm (norm = 1.0)
        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        results.append({
            "embedding": embedding.tolist(),   # بصمة الوجه (512 رقم)
            "bbox": {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
                "normalized": {
                    "x": bbox[0] / w,
                    "y": bbox[1] / h,
                    "width": (bbox[2] - bbox[0]) / w,
                    "height": (bbox[3] - bbox[1]) / h
                }
            },
            "confidence": float(face.det_score)
        })
    return {"faces": results, "count": len(results)}
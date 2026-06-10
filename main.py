import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
import io
import base64

app = FastAPI(title="Face Recognition API", version="1.0")

# CORS للسماح لتطبيقك (Laravel/Django) بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نماذج التحقق من البيانات (Pydantic Models)
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

# تحميل نموذج InsightFace (أدق نموذج متاح)
# ctx_id=0 للـ GPU، -1 للـ CPU
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def read_image(file: UploadFile) -> np.ndarray:
    """قراءة ملف الصورة وتحويلها إلى مصفوفة numpy مع تصحيح الاتجاه تلقائياً (EXIF)"""
    contents = file.file.read()
    try:
        from PIL import Image, ImageOps
        img_pil = Image.open(io.BytesIO(contents))
        img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_np = np.array(img_pil)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to standard OpenCV decode in case of PIL errors
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

def base64_to_image(base64_str: str) -> np.ndarray:
    """تحويل base64 إلى صورة numpy مع تصحيح الاتجاه تلقائياً (EXIF)"""
    # إزالة البادئة إذا وجدت
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    try:
        from PIL import Image, ImageOps
        img_pil = Image.open(io.BytesIO(img_data))
        img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_np = np.array(img_pil)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to standard OpenCV decode in case of PIL errors
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    return img

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

def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """حساب التشابه بين بصمتين"""
    v1 = np.array(emb1)
    v2 = np.array(emb2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# ------------------- نقاط النهاية API -------------------

@app.get("/")
async def root():
    return {"status": "running", "model": "InsightFace (ArcFace)", "accuracy": "99.83%"}

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """كشف جميع الوجوه في الصورة مع إحداثياتها (بدون بصمات)"""
    img = read_image(file)
    faces = face_app.get(img)
    h, w = img.shape[:2]
    result = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        result.append({
            "bbox": {
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "width": bbox[2]-bbox[0], "height": bbox[3]-bbox[1],
                "normalized": {"x": bbox[0]/w, "y": bbox[1]/h, "width": (bbox[2]-bbox[0])/w, "height": (bbox[3]-bbox[1])/h}
            },
            "confidence": float(face.det_score)
        })
    return {"faces": result, "count": len(result)}

@app.post("/detect-embeddings")
async def detect_faces_with_embeddings(file: UploadFile = File(...)):
    """كشف جميع الوجوه مع إحداثياتها وبصماتها"""
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
                "width": bbox[2]-bbox[0], "height": bbox[3]-bbox[1],
                "normalized": {
                    "x": bbox[0]/w, "y": bbox[1]/h,
                    "width": (bbox[2]-bbox[0])/w, "height": (bbox[3]-bbox[1])/h
                }
            },
            "confidence": float(face.det_score)
        })
    return {"faces": results, "count": len(results)}

@app.post("/embedding")
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

@app.post("/embedding-base64")
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

@app.post("/compare")
async def compare_faces(payload: CompareRequest):
    """مقارنة بصمتين (ترسل كـ JSON)"""
    similarity = cosine_similarity(payload.emb1, payload.emb2)
    return {
        "similarity": similarity,
        "is_match": similarity >= payload.threshold,
        "threshold": payload.threshold
    }

@app.post("/compare-files")
async def compare_faces_files(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = 0.65):
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

@app.post("/search")
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

# تشغيل الخدمة
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

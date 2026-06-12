import base64
import io

import cv2
import numpy as np
from fastapi import File, HTTPException, UploadFile


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
    except Exception:
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
    except Exception:
        # Fallback to standard OpenCV decode in case of PIL errors
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    return img


def get_face_coordinates(faces, img_width: int, img_height: int) -> list:
    """استخراج إحداثيات الوجوه (بدون بصمات)"""
    result = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        result.append({
            "bbox": {
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1],
                "normalized": {
                    "x": bbox[0] / img_width,
                    "y": bbox[1] / img_height,
                    "width": (bbox[2] - bbox[0]) / img_width,
                    "height": (bbox[3] - bbox[1]) / img_height
                }
            },
            "confidence": float(face.det_score)
        })
    return result
# ============================================================
# Dockerfile - Face Recognition API
# ============================================================
# مرحلتين: builder لتحميل النماذج، ثم runtime للتشغيل النهائي

FROM python:3.11-slim AS builder

LABEL maintainer="Face Recognition API Team"
LABEL description="Face Recognition API using InsightFace (ArcFace)"

# متغيرات البيئة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# تثبيت مكتبات النظام اللازمة لـ OpenCV و InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# إنشاء المستخدم غير الجذر للتطبيق
RUN groupadd -r faceapi && useradd -r -g faceapi -d /app -s /bin/bash faceapi

# إنشاء المجلدات
WORKDIR /app

# نسخ ملف المتطلبات أولاً (للاستفادة من caching)
COPY requirements.txt .

# تثبيت حزم Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# تحميل نموذج InsightFace (buffalo_l) مسبقاً
# ============================================================
RUN python -c "
import insightface
from insightface.app import FaceAnalysis
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('جارٍ تحميل نموذج InsightFace buffalo_l...')
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
logger.info('✅ تم تحميل النموذج بنجاح')
"

# ============================================================
# المرحلة النهائية (Runtime)
# ============================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8001 \
    HOST=0.0.0.0

# تثبيت مكتبات النظام (ضرورية لتشغيل OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# إنشاء المستخدم غير الجذر
RUN groupadd -r faceapi && useradd -r -g faceapi -d /app -s /bin/bash faceapi

WORKDIR /app

# نسخ الحزم من builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# نسخ النموذج المُحمَّل مسبقاً
COPY --from=builder /root/.insightface /root/.insightface

# نسخ ملفات التطبيق
COPY --chown=faceapi:faceapi main.py .

# التبديل إلى المستخدم غير الجذر
USER faceapi

# فتح المنفذ
EXPOSE 8001

# تشغيل التطبيق عبر uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

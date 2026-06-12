from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import compare_router, detection_router, embedding_router, search_router
from app.core.config import settings

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# CORS للسماح لتطبيقك (Laravel/Django) بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تضمين المسارات (Routers)
app.include_router(detection_router)
app.include_router(embedding_router)
app.include_router(compare_router)
app.include_router(search_router)


@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "InsightFace (ArcFace)",
        "accuracy": "99.83%"
    }
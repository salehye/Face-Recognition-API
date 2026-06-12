from app.api.routes.detection import router as detection_router
from app.api.routes.embedding import router as embedding_router
from app.api.routes.compare import router as compare_router
from app.api.routes.search import router as search_router

__all__ = ["detection_router", "embedding_router", "compare_router", "search_router"]
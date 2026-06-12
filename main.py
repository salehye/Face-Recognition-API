from app.main import app  # noqa: E402, F401

if __name__ == "__main__":
    import uvicorn

    from app.core.config import settings

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
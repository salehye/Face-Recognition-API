class Settings:
    APP_TITLE: str = "Face Recognition API"
    APP_VERSION: str = "1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    MODEL_NAME: str = "buffalo_l"
    DET_SIZE: int = 640
    DEFAULT_THRESHOLD: float = 0.65
    TOP_K: int = 50


settings = Settings()
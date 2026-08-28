from logging import shutdown

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import router
from app.core.config import settings
import kagglehub


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not any(settings.DATASET_DIR.iterdir()):
        kagglehub.dataset_download(
            "yasserh/wine-quality-dataset",
            output_dir=str(settings.DATASET_DIR)
        )

    yield
app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api")
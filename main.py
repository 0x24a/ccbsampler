from __future__ import annotations

import logging
import logging.config
import sys
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cache.manager import CacheManager
from config import load_settings
from models.loader import load_models
from render_queue.gpu_queue import GPUQueue
from routes.resample import router as resample_router
from services.resampler import Renderer

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s (ccbsampler) %(message)s",
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {"handlers": ["default"], "level": "INFO"},
    }
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings("config.yaml")
    logger.info("Loading models...")
    models = load_models(settings)
    cache = CacheManager(max_concurrent_generation=1)
    renderer = Renderer(settings, models, cache)

    gpu_queue = GPUQueue(max_size=settings.performance.max_concurrent_renders)
    gpu_queue.start(renderer.infer)

    app.state.settings = settings
    app.state.renderer = renderer
    app.state.gpu_queue = gpu_queue

    logger.info("Server ready on port %d", settings.port)
    yield

    logger.info("Shutting down...")
    await gpu_queue.stop()


app = FastAPI(title="ccbsampler", lifespan=lifespan)
app.include_router(resample_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": traceback.format_exc()})


@app.get("/")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    settings = load_settings(cfg_path)
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=settings.port,
        log_level="info",
    )

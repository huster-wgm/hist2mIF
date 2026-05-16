"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hist2mif.api.routers import health, jobs
from hist2mif.core.config import settings


def _mount_frontend(application: FastAPI) -> None:
    dist = settings.frontend_dist
    index = dist / "index.html"
    if not dist.is_dir() or not index.is_file():
        return

    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        application.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @application.get("/", include_in_schema=False)
    def spa_index() -> FileResponse:
        return FileResponse(index)

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404)
        return FileResponse(index)


def create_app() -> FastAPI:
    application = FastAPI(title="Hist2mIF API", version="0.2.0")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(jobs.router)

    _mount_frontend(application)
    return application


app = create_app()

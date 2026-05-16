"""Environment-backed settings (resolved once at import)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def repository_root() -> Path:
    """Repository root (contains `pyproject.toml`, `frontend/`)."""
    # src/hist2mif/core/config.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    data_root: Path
    jobs_dir: Path
    max_side: int
    host: str
    port: int
    cors_origins: list[str]
    reload: bool
    frontend_dist: Path


def load_settings() -> Settings:
    data_root = Path(os.environ.get("HIST2MIF_DATA", "data")).resolve()
    cors_raw = os.environ.get("HIST2MIF_CORS", "").strip()
    if cors_raw in {"", "*"}:
        cors = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    else:
        cors = [x.strip() for x in cors_raw.split(",") if x.strip()]

    root = repository_root()
    return Settings(
        data_root=data_root,
        jobs_dir=data_root / "jobs",
        max_side=int(os.environ.get("HIST2MIF_MAX_SIDE", "8192")),
        host=os.environ.get("HIST2MIF_HOST", "0.0.0.0"),
        port=int(os.environ.get("HIST2MIF_PORT", "7860")),
        cors_origins=cors,
        reload=os.environ.get("HIST2MIF_RELOAD", "").lower() in {"1", "true", "yes"},
        frontend_dist=root / "frontend" / "dist",
    )


def _load_dotenv_file() -> None:
    """Load `<repo>/.env` if present; existing OS env vars win (`override=False`)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = repository_root() / ".env"
    if path.is_file():
        load_dotenv(path, override=False)


_load_dotenv_file()
settings = load_settings()

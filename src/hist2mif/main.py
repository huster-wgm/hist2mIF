"""CLI entry: `hist2mif` → Uvicorn."""

from __future__ import annotations

import logging

from hist2mif.core.config import settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(
        "hist2mif.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()

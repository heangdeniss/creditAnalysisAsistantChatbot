"""API gateway entrypoint.

Run:
  uvicorn services.api_gateway:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.api_gateway:app", host="0.0.0.0", port=8000, reload=False)

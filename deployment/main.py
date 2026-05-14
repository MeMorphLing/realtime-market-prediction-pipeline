"""Entry point for the deployment container.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from app.api import create_app

app = create_app()

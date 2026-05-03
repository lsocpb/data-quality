"""Configuration for Tkinter application - DATABASE_URL loading"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _load_database_url_from_env_file(env_path: Path) -> str | None:
    """Try to load DATABASE_URL from .env file"""
    if not env_path.exists():
        return None

    load_dotenv(env_path)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Fallback: parse raw lines looking for connection string
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prefix = line.split("=", 1)[0]
        if "://" in prefix:
            return line

    return None


def get_engine():
    """Create SQLAlchemy engine from DATABASE_URL"""
    database_url = _load_database_url_from_env_file(ENV_PATH)
    
    if not database_url:
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            f"DATABASE_URL not found. Create {ENV_PATH} with: "
            "DATABASE_URL=postgresql://user:pass@localhost/dbname"
        )

    return create_engine(database_url)

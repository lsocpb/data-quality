#!/usr/bin/env python3
"""
Keystroke Dynamics Tkinter Application
========================================

Simple runner for the Tkinter GUI application.

Usage:
    uv run python run_app.py

Requirements:
    - .env file with DATABASE_URL pointing to PostgreSQL
    - All keystroke data must be in the database already
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import main

if __name__ == "__main__":
    main()

"""Entry point for Keystroke Dynamics Tkinter Application"""

import sys
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_engine
from app.ui.main_window import MainWindow


def main():
    try:
        print("Loading database configuration...")
        engine = get_engine()
        
        print("Starting application...")
        app = MainWindow(engine)
        app.mainloop()
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

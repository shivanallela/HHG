import sys
import os

# Add parent directory to path so python can find 'backend' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.factory import create_app

app = create_app()

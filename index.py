import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app
app = create_app()

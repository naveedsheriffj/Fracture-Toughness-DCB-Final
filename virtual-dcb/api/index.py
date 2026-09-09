import sys
import os

# Add parent directory (virtual-dcb) to sys.path so app and prediction_model resolve cleanly
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(base_dir, '..'))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from app import app

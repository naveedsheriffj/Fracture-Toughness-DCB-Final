import sys
import os

# Add virtual-dcb directory to sys.path so modules and packages resolve seamlessly
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(base_dir, '..', 'virtual-dcb'))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from app import app

import os
from .config import load_file

project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
dotenv_path = os.path.join(project_dir, '.env')

load_file(dotenv_path)

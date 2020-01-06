import os
from .config import set_global_seed, load_file, DEFAULT as config

project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
dotenv_path = os.path.join(project_dir, '.env')

load_file(dotenv_path)

__all__ = [set_global_seed, config]

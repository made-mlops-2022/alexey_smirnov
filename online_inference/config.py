import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_path = os.path.join(Path(BASE_DIR).parent, '.env')
load_dotenv(dotenv_path)


path_to_model = os.getenv('MODEL_PATH')
path_to_transformer = os.getenv('TRANSFORMER_PATH')

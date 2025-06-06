from pathlib import Path
from pydantic import BaseSettings

# Rutas base
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

class Settings(BaseSettings):
    # Parámetros de audio
    sample_rate: int = 16_000
    chunk_size: int = 4_096

    # Idiomas por defecto
    src_lang: str = "es"
    tgt_lang: str = "en"

    # Rutas base (se usan más adelante en core/)
    data_dir:   Path = DATA_DIR
    models_dir: Path = MODELS_DIR

    class Config:
        env_prefix = "BABEL_"

settings = Settings()


import os
from dotenv import load_dotenv

load_dotenv()

# Model + embedding names
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_DIR = os.getenv("DB_DIR", "db")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

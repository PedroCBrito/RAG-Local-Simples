"""
Configurações centrais do pipeline RAG Local.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Diretórios base
ROOT_DIR = Path(__file__).resolve().parent.parent

# Suporta tanto 'docs_consultas' quanto 'docs_consulta'
if (ROOT_DIR / "docs_consultas").exists():
    DOCS_DIR = ROOT_DIR / "docs_consultas"
elif (ROOT_DIR / "docs_consulta").exists():
    DOCS_DIR = ROOT_DIR / "docs_consulta"
else:
    DOCS_DIR = ROOT_DIR / "docs_consultas"

FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"
PROMPTS_DIR = ROOT_DIR / "prompts"

# Configurações de Fragmentação (Chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Configurações de Modelos (Ollama / Local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Configurações de Recuperação
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

# Formatos de arquivo suportados
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

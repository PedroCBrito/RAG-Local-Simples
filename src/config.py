"""
Configurações centrais do pipeline RAG Local.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Diretórios base
ROOT_DIR = Path(__file__).resolve().parent.parent


def _project_path(variable_name: str, default_name: str) -> Path:
    """Resolve caminhos relativos do .env a partir da raiz do projeto."""
    configured_path = Path(os.getenv(variable_name, default_name))
    return configured_path if configured_path.is_absolute() else ROOT_DIR / configured_path


def _optional_int(variable_name: str) -> int | None:
    """Lê um inteiro opcional, tratando valor ausente ou vazio como ``None``."""
    raw_value = os.getenv(variable_name)
    return int(raw_value) if raw_value not in (None, "") else None


DOCS_DIR = _project_path("DOCS_DIR", "docs_consulta")
FAISS_INDEX_DIR = _project_path("FAISS_INDEX_DIR", "faiss_index")
PROMPTS_DIR = ROOT_DIR / "prompts"

# Configurações de Fragmentação (Chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Configurações de Modelos (Ollama / Local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))
LLM_NUM_THREAD = _optional_int("LLM_NUM_THREAD")
LLM_NUM_GPU = _optional_int("LLM_NUM_GPU")
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "5m")

# Configurações de Recuperação
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE deve ser maior que zero.")
if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP deve estar entre zero e CHUNK_SIZE - 1.")
if TOP_K_RESULTS <= 0:
    raise ValueError("TOP_K_RESULTS deve ser maior que zero.")
if not 0.0 <= LLM_TEMPERATURE <= 2.0:
    raise ValueError("LLM_TEMPERATURE deve estar entre 0.0 e 2.0.")
if LLM_NUM_CTX <= 0:
    raise ValueError("LLM_NUM_CTX deve ser maior que zero.")
if LLM_NUM_PREDICT <= 0:
    raise ValueError("LLM_NUM_PREDICT deve ser maior que zero.")
if LLM_NUM_THREAD is not None and LLM_NUM_THREAD <= 0:
    raise ValueError("LLM_NUM_THREAD deve ser maior que zero quando informado.")
if LLM_NUM_GPU is not None and LLM_NUM_GPU < 0:
    raise ValueError("LLM_NUM_GPU não pode ser negativo.")

# Formatos de arquivo suportados
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

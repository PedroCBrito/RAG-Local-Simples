"""Configura o provedor local de embeddings usado pelo pipeline RAG."""

import sys
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaEmbeddings

# Permite tanto ``python -m src.embeddings`` quanto ``python src/embeddings.py``.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL


def get_embedding_model(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaEmbeddings:
    """Cria o cliente de embeddings apontando para a instância local do Ollama.

    O objeto retornado implementa a interface ``Embeddings`` do LangChain e pode
    ser passado diretamente ao FAISS na etapa de indexação. O Ollama mantém o
    modelo e executa a vetorização localmente; nenhuma API externa é utilizada.

    Args:
        model_name: Modelo instalado no Ollama. Usa ``EMBEDDING_MODEL_NAME``
            quando não informado.
        base_url: Endereço do Ollama. Usa ``OLLAMA_BASE_URL`` quando não
            informado.
    """
    selected_model = (model_name or EMBEDDING_MODEL_NAME).strip()
    selected_base_url = (base_url or OLLAMA_BASE_URL).strip().rstrip("/")

    if not selected_model:
        raise ValueError("O nome do modelo de embeddings não pode ser vazio.")
    if not selected_base_url:
        raise ValueError("A URL do Ollama não pode ser vazia.")

    return OllamaEmbeddings(
        model=selected_model,
        base_url=selected_base_url,
    )


def validate_embedding_model(embeddings: Optional[OllamaEmbeddings] = None) -> int:
    """Valida a conexão gerando um vetor curto e retorna sua dimensão.

    Raises:
        RuntimeError: Quando o Ollama está indisponível ou o modelo não está
            instalado localmente.
    """
    embedding_model = embeddings or get_embedding_model()

    try:
        vector = embedding_model.embed_query("teste de disponibilidade")
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível gerar embeddings com o Ollama local. "
            f"Confirme que o serviço está ativo e que o modelo "
            f"'{EMBEDDING_MODEL_NAME}' foi instalado com "
            f"'ollama pull {EMBEDDING_MODEL_NAME}'."
        ) from exc

    if not vector:
        raise RuntimeError("O Ollama retornou um vetor de embeddings vazio.")

    return len(vector)


if __name__ == "__main__":
    dimension = validate_embedding_model()
    print(
        f"Modelo de embeddings local '{EMBEDDING_MODEL_NAME}' disponível "
        f"({dimension} dimensões)."
    )

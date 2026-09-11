"""Configuração do modelo de linguagem local executado pelo Ollama."""

import sys
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama

# Permite tanto ``python -m src.llm`` quanto ``python src/llm.py``.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE, OLLAMA_BASE_URL


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
) -> ChatOllama:
    """Cria o cliente da LLM local com temperatura baixa e determinística."""
    selected_model = (model_name or LLM_MODEL_NAME).strip()
    selected_base_url = (base_url or OLLAMA_BASE_URL).strip().rstrip("/")
    selected_temperature = (
        LLM_TEMPERATURE if temperature is None else float(temperature)
    )

    if not selected_model:
        raise ValueError("O nome do modelo da LLM não pode ser vazio.")
    if not selected_base_url:
        raise ValueError("A URL do Ollama não pode ser vazia.")
    if not 0.0 <= selected_temperature <= 2.0:
        raise ValueError("A temperatura da LLM deve estar entre 0.0 e 2.0.")

    return ChatOllama(
        model=selected_model,
        temperature=selected_temperature,
        base_url=selected_base_url,
    )


def validate_llm(llm: Optional[ChatOllama] = None) -> str:
    """Executa uma inferência curta para validar serviço, modelo e cliente."""
    local_llm = llm or get_llm()

    try:
        response = local_llm.invoke(
            "Responda apenas com a palavra OK, sem pontuação ou explicações."
        )
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível executar a LLM no Ollama local. "
            f"Confirme que o serviço está ativo e que o modelo "
            f"'{LLM_MODEL_NAME}' foi instalado com "
            f"'ollama pull {LLM_MODEL_NAME}'."
        ) from exc

    content = str(response.content).strip()
    if not content:
        raise RuntimeError("A LLM local retornou uma resposta vazia.")
    return content


if __name__ == "__main__":
    answer = validate_llm()
    print(
        f"LLM local '{LLM_MODEL_NAME}' disponível "
        f"(temperatura={LLM_TEMPERATURE}): {answer}"
    )

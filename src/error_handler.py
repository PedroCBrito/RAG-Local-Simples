"""Tradução centralizada de falhas técnicas em mensagens amigáveis."""

import pickle
import sys
from dataclasses import dataclass
from typing import Optional

import httpx

try:
    from ollama import ResponseError as OllamaResponseError
except ImportError:  # Permite informar a dependência ausente de forma amigável.
    OllamaResponseError = None


@dataclass(frozen=True)
class FriendlyError:
    """Mensagem segura para apresentação no terminal."""

    title: str
    detail: str
    suggestion: str

    def render(self) -> str:
        return (
            f"❌ {self.title}\n"
            f"   {self.detail}\n"
            f"💡 Como resolver: {self.suggestion}"
        )


def _exception_chain(error: BaseException):
    """Percorre causas encadeadas sem entrar em ciclos."""
    current: Optional[BaseException] = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def classify_error(error: BaseException, operation: str = "consulta") -> FriendlyError:
    """Classifica uma exceção e fornece uma orientação adequada ao usuário."""
    chain = list(_exception_chain(error))
    combined_message = " ".join(str(item) for item in chain).lower()

    if any(isinstance(item, ModuleNotFoundError) for item in chain):
        return FriendlyError(
            "Dependência Python ausente.",
            "Um pacote necessário para executar o RAG não está instalado.",
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        )

    if any(isinstance(item, FileNotFoundError) for item in chain):
        return FriendlyError(
            "Índice vetorial não encontrado.",
            "Os arquivos index.faiss e index.pkl ainda não estão disponíveis.",
            "adicione documentos em docs_consulta/ e execute "
            ".\\.venv\\Scripts\\python.exe src\\ingest.py",
        )

    if "nenhum documento" in combined_message or "diretório" in combined_message and (
        "não existe" in combined_message or "vazio" in combined_message
    ):
        return FriendlyError(
            "Nenhum documento disponível para ingestão.",
            "A pasta docs_consulta/ não existe, está vazia ou não contém PDF/TXT válido.",
            "adicione pelo menos um arquivo .pdf ou .txt em docs_consulta/ e execute a ingestão.",
        )

    corruption_terms = ("faiss", "pickle", "deserial", "corrupt", "incomplete", "incompleto")
    if any(isinstance(item, (pickle.UnpicklingError, EOFError)) for item in chain) or (
        any(term in combined_message for term in corruption_terms)
        and any(term in combined_message for term in ("error", "erro", "fail", "falha", "invalid"))
    ):
        return FriendlyError(
            "Índice vetorial corrompido ou incompatível.",
            "Não foi possível carregar com segurança os arquivos do FAISS.",
            "remova o índice inválido e gere-o novamente executando src\\ingest.py.",
        )

    if OllamaResponseError is not None and any(
        isinstance(item, OllamaResponseError) for item in chain
    ):
        if "not found" in combined_message or "404" in combined_message:
            return FriendlyError(
                "Modelo local não encontrado no Ollama.",
                "O modelo configurado ainda não foi baixado.",
                "execute 'ollama pull llama3.2:3b' e tente novamente.",
            )
        return FriendlyError(
            "O Ollama recusou a solicitação.",
            "O serviço respondeu com erro ao processar a requisição.",
            "confira os modelos com 'ollama list' e reinicie o serviço se necessário.",
        )

    if any(isinstance(item, httpx.TimeoutException) for item in chain):
        return FriendlyError(
            "Tempo limite excedido ao acessar o Ollama.",
            "O modelo local demorou além do esperado para responder.",
            "confirme os recursos disponíveis e tente novamente; a primeira execução pode ser mais lenta.",
        )

    if any(isinstance(item, httpx.ConnectError) for item in chain) or any(
        phrase in combined_message
        for phrase in ("connection refused", "connect error", "failed to connect", "winerror 10061")
    ):
        return FriendlyError(
            "Serviço Ollama indisponível.",
            "Não foi possível conectar a http://localhost:11434.",
            "inicie o aplicativo Ollama ou execute 'ollama serve' e tente novamente.",
        )

    if isinstance(error, (TypeError, ValueError)):
        return FriendlyError(
            "Entrada inválida.",
            str(error) or "A pergunta ou configuração informada não é válida.",
            "informe uma pergunta textual não vazia e tente novamente.",
        )

    return FriendlyError(
        f"Não foi possível concluir a {operation}.",
        str(error) or "Ocorreu uma falha inesperada.",
        "verifique o índice, o serviço Ollama e tente novamente.",
    )


def print_friendly_error(
    error: BaseException | FriendlyError,
    operation: str = "consulta",
    file=None,
) -> None:
    """Exibe uma falha amigável, sem expor traceback no uso normal do CLI."""
    output = file if file is not None else sys.stderr
    friendly_error = (
        error if isinstance(error, FriendlyError) else classify_error(error, operation)
    )
    print(friendly_error.render(), file=output)

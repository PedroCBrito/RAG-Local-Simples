"""Recuperação semântica de trechos armazenados no índice FAISS local."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Permite tanto ``python -m src.retriever`` quanto execução direta do arquivo.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FAISS_INDEX_DIR, TOP_K_RESULTS
from src.ingest import load_vector_store


def get_retriever(
    k: Optional[int] = None,
    index_dir: Path | str = FAISS_INDEX_DIR,
    vector_store: Optional[FAISS] = None,
) -> BaseRetriever:
    """Cria um retriever de similaridade sobre o índice FAISS.

    Quando ``vector_store`` não é informado, o índice é recarregado do disco
    sem reler ou fragmentar os documentos originais.
    """
    effective_k = TOP_K_RESULTS if k is None else k
    if isinstance(effective_k, bool) or not isinstance(effective_k, int):
        raise TypeError("O parâmetro 'k' deve ser um número inteiro.")
    if effective_k <= 0:
        raise ValueError("O parâmetro 'k' deve ser maior que zero.")

    store = vector_store or load_vector_store(index_dir=index_dir)
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": effective_k},
    )


def retrieve_documents(
    query: str,
    k: Optional[int] = None,
    index_dir: Path | str = FAISS_INDEX_DIR,
    vector_store: Optional[FAISS] = None,
) -> list[Document]:
    """Retorna os até ``k`` trechos mais próximos de uma query textual."""
    if not isinstance(query, str):
        raise TypeError("A query deve ser uma string.")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("A query não pode ser vazia.")

    retriever = get_retriever(
        k=k,
        index_dir=index_dir,
        vector_store=vector_store,
    )
    return retriever.invoke(normalized_query)


def main() -> None:
    """Executa uma busca semântica no índice através da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Retorna os trechos mais próximos do índice FAISS local."
    )
    parser.add_argument("query", help="Texto usado na busca semântica.")
    parser.add_argument(
        "-k",
        type=int,
        default=TOP_K_RESULTS,
        help=f"Número máximo de resultados (padrão: {TOP_K_RESULTS}).",
    )
    args = parser.parse_args()

    try:
        documents = retrieve_documents(args.query, k=args.k)
    except Exception as exc:
        print(f"❌ Falha na recuperação: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not documents:
        print("Nenhum trecho encontrado.")
        return

    print(f"🔎 {len(documents)} trecho(s) encontrado(s):")
    for position, document in enumerate(documents, start=1):
        source = document.metadata.get("filename") or document.metadata.get(
            "source", "fonte desconhecida"
        )
        page = document.metadata.get("page")
        page_info = f", página {page + 1}" if isinstance(page, int) else ""
        preview = " ".join(document.page_content.split())
        print(f"\n[{position}] {source}{page_info}")
        print(preview)


if __name__ == "__main__":
    main()

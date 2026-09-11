"""Constrói, persiste e recarrega o índice vetorial FAISS local."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Permite executar diretamente com ``python src/ingest.py``.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DOCS_DIR, FAISS_INDEX_DIR
from src.embeddings import get_embedding_model
from src.loader import scan_and_load_documents
from src.splitter import split_documents

INDEX_NAME = "index"


def index_exists(index_dir: Path | str = FAISS_INDEX_DIR) -> bool:
    """Retorna ``True`` quando os dois arquivos necessários do FAISS existem."""
    target_dir = Path(index_dir)
    return (
        (target_dir / f"{INDEX_NAME}.faiss").is_file()
        and (target_dir / f"{INDEX_NAME}.pkl").is_file()
    )


def build_vector_store(
    chunks: Sequence[Document],
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """Gera os embeddings dos chunks e constrói um índice FAISS em memória."""
    if not chunks:
        raise ValueError("Não há chunks para adicionar ao índice vetorial.")

    embedding_model = embeddings or get_embedding_model()
    return FAISS.from_documents(list(chunks), embedding_model)


def save_vector_store(
    vector_store: FAISS,
    index_dir: Path | str = FAISS_INDEX_DIR,
) -> None:
    """Persiste o índice e os metadados no diretório configurado."""
    target_dir = Path(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_dir), index_name=INDEX_NAME)

    if not index_exists(target_dir):
        raise RuntimeError(f"Falha ao persistir o índice FAISS em '{target_dir}'.")


def load_vector_store(
    index_dir: Path | str = FAISS_INDEX_DIR,
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """Recarrega o índice salvo sem reler nem fragmentar os documentos.

    O arquivo ``index.pkl`` é desserializado porque foi produzido localmente por
    este projeto. Não substitua esses arquivos por conteúdo de origem não
    confiável.
    """
    target_dir = Path(index_dir)
    if not index_exists(target_dir):
        raise FileNotFoundError(
            f"Índice FAISS ausente ou incompleto em '{target_dir}'. "
            "Execute 'python src/ingest.py' para criá-lo."
        )

    embedding_model = embeddings or get_embedding_model()
    return FAISS.load_local(
        str(target_dir),
        embedding_model,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def ingest_documents(
    documents_dir: Path | str = DOCS_DIR,
    index_dir: Path | str = FAISS_INDEX_DIR,
    embeddings: Optional[Embeddings] = None,
) -> FAISS:
    """Executa carga, chunking, vetorização e persistência do índice."""
    documents = scan_and_load_documents(documents_dir)
    if not documents:
        raise RuntimeError(
            f"Nenhum documento válido foi carregado de '{Path(documents_dir)}'."
        )

    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("A fragmentação não produziu chunks para indexação.")

    print(f"🧠 Gerando embeddings locais para {len(chunks)} chunk(s)...")
    vector_store = build_vector_store(chunks, embeddings=embeddings)

    print(f"💾 Salvando índice vetorial em '{Path(index_dir)}'...")
    save_vector_store(vector_store, index_dir=index_dir)
    print(
        f"✅ Índice FAISS criado: {len(chunks)} vetor(es), "
        f"arquivos '{INDEX_NAME}.faiss' e '{INDEX_NAME}.pkl'."
    )
    return vector_store


def main() -> None:
    """Interface de linha de comando para criar ou validar o reload do índice."""
    parser = argparse.ArgumentParser(
        description="Cria ou recarrega o índice FAISS do RAG local."
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Apenas recarrega o índice existente, sem processar os documentos.",
    )
    args = parser.parse_args()

    try:
        if args.load_only:
            vector_store = load_vector_store()
            total_vectors = vector_store.index.ntotal
            print(
                f"✅ Índice FAISS recarregado da memória local "
                f"({total_vectors} vetor(es))."
            )
        else:
            ingest_documents()
    except Exception as exc:
        print(f"❌ Falha na ingestão: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

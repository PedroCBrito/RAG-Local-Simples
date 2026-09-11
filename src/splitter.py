"""
Módulo de fragmentação de texto (Chunking) para o pipeline RAG Local.
Utiliza RecursiveCharacterTextSplitter garantindo a preservação e enriquecimento dos metadados.
"""

import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Garante acesso aos módulos do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Instancia o RecursiveCharacterTextSplitter configurado com parâmetros padrão ou customizados.
    
    Args:
        chunk_size: Tamanho máximo de caracteres por chunk (padrão definido em src/config.py).
        chunk_overlap: Quantidade de sobreposição de caracteres entre chunks adjacentes.
        separators: Lista ordenada de separadores para divisão textual.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    if effective_chunk_size <= 0:
        raise ValueError("O 'chunk_size' deve ser maior que zero.")
    if effective_chunk_overlap < 0:
        raise ValueError("O 'chunk_overlap' não pode ser negativo.")
    if effective_chunk_overlap >= effective_chunk_size:
        raise ValueError("O 'chunk_overlap' deve ser estritamente menor que o 'chunk_size'.")

    default_separators = separators or ["\n\n", "\n", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=default_separators
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Divide uma lista de Documentos em chunks menores, preservando todos os metadados
    originais e adicionando índices de rastreamento para cada chunk.
    
    Args:
        documents: Lista de documentos brutos carregados pelo loader.
        chunk_size: Tamanho do chunk customizado (opcional).
        chunk_overlap: Sobreposição de chunk customizada (opcional).
        
    Returns:
        Lista de documentos fragmentados (chunks) com metadados preservados.
    """
    if not documents:
        print("ℹ️  Nenhum documento fornecido para fragmentação.")
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print(f"✂️  Iniciando fragmentação de {len(documents)} documento(s)/página(s)...")
    print(f"   ⚙️  Configuração: chunk_size={splitter._chunk_size}, chunk_overlap={splitter._chunk_overlap}")

    raw_chunks = splitter.split_documents(documents)
    enriched_chunks: List[Document] = []

    # Dicionário para controlar indexação sequencial de chunks por documento de origem
    doc_chunk_counters = {}

    for chunk in raw_chunks:
        # Recupera chave única do documento de origem (filename ou source)
        source_key = chunk.metadata.get("source") or chunk.metadata.get(
            "filename", "unknown"
        )
        current_index = doc_chunk_counters.get(source_key, 0)
        doc_chunk_counters[source_key] = current_index + 1

        # Cria uma cópia preservando os metadados anteriores e enriquecendo
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = current_index
        metadata["chunk_char_count"] = len(chunk.page_content)

        enriched_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )

    print(f"✅ Fragmentação concluída com sucesso: {len(enriched_chunks)} chunks gerados.\n")
    return enriched_chunks


if __name__ == "__main__":
    # Teste de execução rápida com documentos fictícios
    sample_docs = [
        Document(
            page_content=(
                "O pipeline de RAG local permite consulta semântica privada e offline. " * 30
            ),
            metadata={"source": "manual_rag.pdf", "filename": "manual_rag.pdf", "page": 0}
        )
    ]
    
    chunks = split_documents(sample_docs)
    for i, c in enumerate(chunks[:3]):
        print(f"--- Chunk #{i + 1} ---")
        print(f"Metadados: {c.metadata}")
        print(f"Tamanho: {len(c.page_content)} caracteres")
        print(f"Prévia: {c.page_content[:100]}...\n")

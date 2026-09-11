"""Formatação legível das respostas e fontes do RAG para o terminal."""

from pathlib import Path
from typing import Any, Mapping, Sequence


def _source_fields(source: Any, structured: bool) -> tuple[str, int | None, int | None]:
    """Extrai arquivo, página e chunk de um Document ou fonte estruturada."""
    if structured:
        metadata = source if isinstance(source, Mapping) else {}
        raw_source = metadata.get("filename") or "Documento desconhecido"
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
    else:
        metadata = getattr(source, "metadata", {})
        raw_source = metadata.get("filename") or metadata.get(
            "source", "Documento desconhecido"
        )
        raw_page = metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else None
        chunk_index = metadata.get("chunk_index")

    filename = Path(str(raw_source)).name
    return (
        filename,
        page if isinstance(page, int) else None,
        chunk_index if isinstance(chunk_index, int) else None,
    )


def format_sources(sources: Sequence[Any], structured: bool = False) -> str:
    """Formata fontes como uma lista com bullets, removendo duplicatas."""
    if not sources:
        return "- Nenhuma fonte consultada."

    lines = []
    seen = set()
    for source in sources:
        filename, page, chunk_index = _source_fields(source, structured)
        source_key = (filename, page, chunk_index)
        if source_key in seen:
            continue
        seen.add(source_key)

        extension = Path(filename).suffix.lower()
        if extension == ".pdf" and page is not None:
            detail = f" (Página {page})"
        elif chunk_index is not None:
            detail = f" (Trecho {chunk_index + 1})"
        elif page is not None:
            detail = f" (Página {page})"
        else:
            detail = ""

        lines.append(f"- {filename}{detail}")

    return "\n".join(lines) if lines else "- Nenhuma fonte consultada."


def format_rag_output(result: Any) -> str:
    """Gera o bloco final de resposta e fontes apresentado no terminal."""
    if isinstance(result, Mapping):
        answer = result.get("answer") or result.get("result") or "Resposta vazia."
        structured_sources = result.get("sources")
        if structured_sources is not None:
            sources = structured_sources
            sources_text = format_sources(sources, structured=True)
        else:
            sources = result.get("source_documents") or result.get("context") or []
            sources_text = format_sources(sources)
    else:
        answer = str(result)
        sources_text = format_sources([])

    return f"[Resposta]: {str(answer).strip()}\n\n[Fontes Consultadas]:\n{sources_text}"

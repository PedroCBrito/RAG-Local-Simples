"""
Módulo de varredura e carga de documentos para o RAG Local.
Processa arquivos nos formatos .pdf e .txt a partir da pasta de consultas.
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)
from langchain_community.document_loaders import PyPDFLoader

# Garante acesso aos módulos do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.config import DOCS_DIR, SUPPORTED_EXTENSIONS
except ImportError:
    DOCS_DIR = ROOT_DIR / "docs_consultas" if (ROOT_DIR / "docs_consultas").exists() else ROOT_DIR / "docs_consulta"
    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_pdf_file(file_path: Path) -> List[Document]:
    """
    Carrega arquivo PDF extraindo conteúdo por página com metadados.
    """
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = file_path.name
            doc.metadata["extension"] = ".pdf"
            doc.metadata["file_size"] = file_path.stat().st_size
        return docs
    except Exception as e:
        print(f"⚠️  [Erro]: Falha ao processar arquivo PDF '{file_path.name}': {e}")
        return []


def load_txt_file(file_path: Path) -> List[Document]:
    """
    Carrega arquivo TXT com suporte resiliente a múltiplos encodings (UTF-8, Latin-1, CP1252).
    """
    # CP1252 vem antes de Latin-1 porque ambos aceitam quase todos os bytes,
    # mas CP1252 preserva corretamente aspas e travessões comuns no Windows.
    encodings = ["utf-8", "cp1252", "latin-1"]
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "extension": ".txt",
                        "file_size": file_path.stat().st_size,
                    },
                )
            ]
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️  [Erro]: Falha ao ler arquivo de texto '{file_path.name}': {e}")
            return []
            
    print(f"⚠️  [Erro]: Não foi possível decodificar o arquivo '{file_path.name}' com encodings conhecidos.")
    return []


def load_single_document(file_path: Path) -> List[Document]:
    """
    Identifica o formato do arquivo e direciona para o carregador correspondente.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return load_pdf_file(file_path)
    elif ext == ".txt":
        return load_txt_file(file_path)
    return []


def scan_and_load_documents(directory_path: Optional[Path | str] = None) -> List[Document]:
    """
    Varre a pasta de documentos (.pdf e .txt) e retorna a lista de documentos carregados.
    """
    target_dir = Path(directory_path) if directory_path else DOCS_DIR

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"⚠️  [Aviso]: Diretório '{target_dir}' não existe.")
        return []

    print(f"📂 Iniciando varredura em: '{target_dir}'")
    
    # Busca arquivos suportados de forma recursiva
    found_files = sorted(
        (
            f
            for f in target_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )

    if not found_files:
        print(f"ℹ️  Nenhum documento ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) encontrado em '{target_dir}'.")
        return []

    print(f"📄 Encontrado(s) {len(found_files)} arquivo(s):")
    for file in found_files:
        print(f"   • {file.name} ({file.stat().st_size} bytes)")

    loaded_docs: List[Document] = []
    for file_path in found_files:
        docs = load_single_document(file_path)
        if docs:
            loaded_docs.extend(docs)

    print(f"✅ Varredura e carga finalizada: {len(loaded_docs)} página(s)/seção(ões) extraída(s).\n")
    return loaded_docs


if __name__ == "__main__":
    # Teste de execução direta do módulo
    documents = scan_and_load_documents()
    print(f"Total de objetos carregados: {len(documents)}")

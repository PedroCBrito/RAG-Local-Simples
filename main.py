"""
Ponto de entrada e interface CLI para o RAG Local.
"""

import os
import sys
import argparse
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para garantir imports relativos de src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DOCS_DIR, FAISS_INDEX_DIR, SUPPORTED_EXTENSIONS
from src.error_handler import FriendlyError, print_friendly_error
from src.ingest import ingest_documents
from src.output_formatter import format_rag_output, format_sources as _format_sources


def print_banner() -> None:
    """Exibe o banner inicial da aplicação."""
    print("=" * 60)
    print("       🤖 RAG Local - Consulta Semântica de Documentos       ")
    print("=" * 60)
    print("Comandos disponíveis:")
    print("  - Digite sua pergunta e pressione Enter")
    print("  - 'sair' ou 'exit'  : Encerrar a aplicação")
    print("  - 'limpar' ou 'clear': Limpar a tela")
    print("  - 'ajuda' ou 'help'  : Exibir esta mensagem de ajuda")
    print("=" * 60 + "\n")


def check_prerequisites() -> bool:
    """
    Verifica se os pré-requisitos mínimos para consulta estão atendidos.
    Retorna True se o índice FAISS existir, False caso contrário.
    """
    index_file = FAISS_INDEX_DIR / "index.faiss"
    metadata_file = FAISS_INDEX_DIR / "index.pkl"
    if not (index_file.is_file() and metadata_file.is_file()):
        has_documents = DOCS_DIR.is_dir() and any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in DOCS_DIR.rglob("*")
        )
        if not has_documents:
            error = FriendlyError(
                "Índice vetorial ausente e pasta de documentos vazia.",
                f"Nenhum arquivo PDF ou TXT foi encontrado em '{DOCS_DIR.name}/'.",
                f"adicione documentos em '{DOCS_DIR.name}/' e execute "
                ".\\.venv\\Scripts\\python.exe src\\ingest.py",
            )
        else:
            error = FileNotFoundError("Índice FAISS ausente ou incompleto.")
        print_friendly_error(error, operation="inicialização", file=sys.stdout)
        return False

    if index_file.stat().st_size == 0 or metadata_file.stat().st_size == 0:
        error = FriendlyError(
            "Índice vetorial corrompido ou incompleto.",
            "Um ou mais arquivos do índice estão vazios.",
            "gere novamente o índice executando "
            ".\\.venv\\Scripts\\python.exe src\\ingest.py",
        )
        print_friendly_error(error, operation="inicialização", file=sys.stdout)
        return False
    return True


def format_sources(source_docs: list) -> str:
    """Mantém compatibilidade com chamadas existentes do formatador de fontes."""
    return _format_sources(source_docs)


def run_single_query(rag_chain, query: str) -> bool:
    """Executa uma consulta, exibe o resultado e informa se houve sucesso."""
    query = query.strip()
    if not query:
        return False

    print("\n🔍 Buscando contexto e gerando resposta...")
    try:
        result = rag_chain.invoke(query)
        print("\n" + "-" * 60)
        print(format_rag_output(result))
        print("-" * 60)
        return True

    except Exception as error:
        print()
        print_friendly_error(error, operation="consulta", file=sys.stdout)
        return False


def interactive_cli(rag_chain) -> None:
    """Loop interativo no terminal para conversação com a base de dados."""
    print_banner()

    while True:
        try:
            user_input = input("Pergunta: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["sair", "exit", "quit", "q"]:
                print("\n👋 Encerrando RAG Local. Até logo!")
                break

            if user_input.lower() in ["limpar", "clear", "cls"]:
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue

            if user_input.lower() in ["ajuda", "help", "h"]:
                print_banner()
                continue

            run_single_query(rag_chain, user_input)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Operação cancelada pelo usuário. Encerrando...")
            break


def main() -> None:
    """Função principal que orquestra o carregamento e execução do CLI."""
    parser = argparse.ArgumentParser(description="Interface CLI para o RAG Local")
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Executa uma pergunta direta via linha de comando sem abrir o modo interativo."
    )
    args = parser.parse_args()

    print("⏳ Atualizando o índice com os documentos locais...")
    try:
        ingest_documents()
    except Exception as error:
        print_friendly_error(error, operation="ingestão", file=sys.stdout)
        sys.exit(1)

    # Confirma que a ingestão produziu os dois arquivos necessários.
    if not check_prerequisites():
        sys.exit(1)

    print("⏳ Carregando modelos e índice vetorial...")
    try:
        from src.rag_chain import get_rag_chain
        rag_chain = get_rag_chain()
        print("✅ Sistema pronto para consultas!\n")
    except Exception as error:
        print_friendly_error(error, operation="inicialização", file=sys.stdout)
        sys.exit(1)

    if args.query:
        if not run_single_query(rag_chain, args.query):
            sys.exit(1)
    else:
        interactive_cli(rag_chain)


if __name__ == "__main__":
    main()

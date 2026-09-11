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

# Constantes e caminhos padrão
FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"
DOCS_DIR = ROOT_DIR / "docs_consulta"


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
    pkl_file = FAISS_INDEX_DIR / "index.pkl"

    if not (index_file.exists() and pkl_file.exists()):
        print("⚠️  [Aviso]: Índice vetorial não encontrado em 'faiss_index/'.")
        print(f"👉 Certifique-se de adicionar documentos em '{DOCS_DIR.name}/' e executar a ingestão:")
        print("   python src/ingest.py\n")
        return False
    return True


def format_sources(source_docs: list) -> str:
    """Formata a lista de documentos e trechos retornados pelo retriever."""
    if not source_docs:
        return "Nenhuma fonte encontrada."

    formatted = []
    seen = set()
    
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source", "Documento desconhecido")
        page = doc.metadata.get("page", None)
        filename = Path(source).name

        doc_key = (filename, page)
        if doc_key in seen:
            continue
        seen.add(doc_key)

        page_info = f" (Página {page + 1})" if page is not None else ""
        formatted.append(f"  [{len(seen)}] {filename}{page_info}")

    return "\n".join(formatted)


def run_single_query(rag_chain, query: str) -> None:
    """Executa uma única consulta e exibe a resposta com as fontes."""
    query = query.strip()
    if not query:
        return

    print("\n🔍 Buscando contexto e gerando resposta...")
    try:
        # Suporta tanto chamadas com retorno direto quanto dicionários (invoke)
        result = rag_chain.invoke(query)

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("result") or str(result)
            source_docs = result.get("source_documents") or result.get("context") or []
        else:
            answer = str(result)
            source_docs = []

        print("\n" + "-" * 50)
        print("📢 [Resposta]:")
        print(answer)
        print("-" * 50)

        if source_docs:
            print("\n📚 [Fontes Consultadas]:")
            print(format_sources(source_docs))
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ Erro durante a inferência: {e}")
        print("💡 Dica: Verifique se o serviço do Ollama está em execução (ex.: 'ollama serve').")


def interactive_cli(rag_chain) -> None:
    """Loop interativo no terminal para conversação com a base de dados."""
    print_banner()

    while True:
        try:
            user_input = input("💬 Pergunta: ").strip()

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

    # Verifica se os arquivos do índice FAISS existem
    if not check_prerequisites():
        sys.exit(1)

    print("⏳ Carregando modelos e índice vetorial...")
    try:
        from src.rag_chain import get_rag_chain
        rag_chain = get_rag_chain()
        print("✅ Sistema pronto para consultas!\n")
    except ImportError as e:
        print(f"❌ Erro ao importar cadeia RAG de 'src.rag_chain': {e}")
        print("💡 Certifique-se de que os módulos dentro de 'src/' foram implementados.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Falha ao inicializar a cadeia RAG: {e}")
        print("💡 Certifique-se de que o Ollama está rodando e os modelos estão instalados.")
        sys.exit(1)

    if args.query:
        run_single_query(rag_chain, args.query)
    else:
        interactive_cli(rag_chain)


if __name__ == "__main__":
    main()

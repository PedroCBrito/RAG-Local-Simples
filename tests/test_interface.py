import pickle
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import httpx
from langchain_core.documents import Document

import main
from src.error_handler import classify_error
from src.output_formatter import format_rag_output, format_sources


class OutputFormatterTests(unittest.TestCase):
    def test_structured_output_is_legible_and_deduplicated(self):
        result = {
            "answer": "Resposta local.",
            "sources": [
                {"filename": "doc.pdf", "page": 2, "chunk_index": 0},
                {"filename": "notas.txt", "page": None, "chunk_index": 0},
                {"filename": "notas.txt", "page": None, "chunk_index": 0},
            ],
        }
        output = format_rag_output(result)
        self.assertIn("[Resposta]: Resposta local.", output)
        self.assertIn("- doc.pdf (Página 2)", output)
        self.assertEqual(1, output.count("notas.txt"))
        self.assertIn("- notas.txt (Trecho 1)", output)

    def test_document_sources_and_empty_sources(self):
        document = Document(
            page_content="texto",
            metadata={"filename": "fonte.txt", "chunk_index": 1},
        )
        self.assertEqual("- fonte.txt (Trecho 2)", format_sources([document]))
        self.assertEqual("- Nenhuma fonte consultada.", format_sources([]))


class ErrorHandlerTests(unittest.TestCase):
    def test_common_errors_are_classified(self):
        cases = [
            (FileNotFoundError("index"), "Índice vetorial não encontrado"),
            (pickle.UnpicklingError("invalid"), "corrompido"),
            (httpx.ConnectError("connection refused"), "Ollama indisponível"),
            (httpx.ReadTimeout("timeout"), "Tempo limite"),
            (ValueError("query vazia"), "Entrada inválida"),
        ]
        for error, title in cases:
            with self.subTest(error=error):
                self.assertIn(title, classify_error(error).title)


class CliTests(unittest.TestCase):
    def test_query_success_and_failure_return_status(self):
        success_chain = unittest.mock.Mock()
        success_chain.invoke.return_value = {"answer": "OK", "sources": []}
        failure_chain = unittest.mock.Mock()
        failure_chain.invoke.side_effect = httpx.ConnectError("connection refused")

        with redirect_stdout(StringIO()):
            self.assertTrue(main.run_single_query(success_chain, "pergunta"))
            self.assertFalse(main.run_single_query(failure_chain, "pergunta"))
            self.assertFalse(main.run_single_query(success_chain, "   "))

    def test_interactive_cli_accepts_multiple_queries_and_exit(self):
        chain = unittest.mock.Mock()
        chain.invoke.return_value = {"answer": "OK", "sources": []}
        with patch("builtins.input", side_effect=["uma", "duas", "sair"]):
            with redirect_stdout(StringIO()):
                main.interactive_cli(chain)

        self.assertEqual(2, chain.invoke.call_count)
        self.assertEqual(["uma", "duas"], [call.args[0] for call in chain.invoke.call_args_list])

    def test_main_rebuilds_index_before_direct_query(self):
        fake_chain = unittest.mock.Mock()
        with patch("sys.argv", ["main.py", "-q", "pergunta"]), patch.object(
            main, "ingest_documents"
        ) as ingest, patch.object(
            main, "check_prerequisites", return_value=True
        ), patch(
            "src.rag_chain.get_rag_chain", return_value=fake_chain
        ), patch.object(
            main, "run_single_query", return_value=True
        ) as run_query, redirect_stdout(StringIO()):
            main.main()

        ingest.assert_called_once_with()
        run_query.assert_called_once_with(fake_chain, "pergunta")

    def test_main_stops_when_automatic_ingestion_fails(self):
        with patch("sys.argv", ["main.py"]), patch.object(
            main, "ingest_documents", side_effect=RuntimeError("Nenhum documento")
        ), patch.object(main, "check_prerequisites") as prerequisites, redirect_stdout(
            StringIO()
        ):
            with self.assertRaises(SystemExit) as exit_context:
                main.main()

        self.assertEqual(1, exit_context.exception.code)
        prerequisites.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from src.loader import scan_and_load_documents
from src.splitter import get_text_splitter, split_documents


class LoaderTests(unittest.TestCase):
    def test_scan_loads_supported_encodings_and_ignores_other_files(self):
        with TemporaryDirectory() as temporary_dir:
            docs_dir = Path(temporary_dir)
            (docs_dir / "b_utf8.txt").write_text("Conteúdo UTF-8", encoding="utf-8")
            (docs_dir / "a_latin1.txt").write_text("Conteúdo Latin-1", encoding="latin-1")
            (docs_dir / "ignored.md").write_text("Ignorado", encoding="utf-8")

            with redirect_stdout(StringIO()):
                documents = scan_and_load_documents(docs_dir)

        self.assertEqual(2, len(documents))
        self.assertEqual(
            ["a_latin1.txt", "b_utf8.txt"],
            [document.metadata["filename"] for document in documents],
        )
        for document in documents:
            self.assertEqual(".txt", document.metadata["extension"])
            self.assertGreater(document.metadata["file_size"], 0)
            self.assertIn("source", document.metadata)

    def test_missing_directory_returns_empty_list(self):
        with TemporaryDirectory() as temporary_dir:
            missing = Path(temporary_dir) / "missing"
            with redirect_stdout(StringIO()):
                self.assertEqual([], scan_and_load_documents(missing))


class SplitterTests(unittest.TestCase):
    def test_split_preserves_and_enriches_metadata(self):
        source = Document(
            page_content=("Python e FAISS formam um pipeline local. " * 20).strip(),
            metadata={"source": "manual.txt", "filename": "manual.txt"},
        )
        with redirect_stdout(StringIO()):
            chunks = split_documents([source], chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(list(range(len(chunks))), [c.metadata["chunk_index"] for c in chunks])
        for chunk in chunks:
            self.assertEqual("manual.txt", chunk.metadata["filename"])
            self.assertEqual(len(chunk.page_content), chunk.metadata["chunk_char_count"])
            self.assertLessEqual(len(chunk.page_content), 120)

    def test_invalid_splitter_parameters_are_rejected(self):
        for chunk_size, overlap in ((0, 0), (100, -1), (100, 100)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    get_text_splitter(chunk_size, overlap)


if __name__ == "__main__":
    unittest.main()

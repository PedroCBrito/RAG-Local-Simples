import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from src.ingest import (
    build_vector_store,
    index_exists,
    load_vector_store,
    save_vector_store,
)
from src.retriever import get_retriever, retrieve_documents
from tests.fakes import KeywordEmbeddings


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = KeywordEmbeddings()
        self.documents = [
            Document(page_content="Python executa o sistema.", metadata={"filename": "python.txt"}),
            Document(page_content="FAISS armazena os vetores.", metadata={"filename": "faiss.txt"}),
            Document(page_content="Café é uma bebida.", metadata={"filename": "cafe.txt"}),
        ]

    def test_build_save_reload_and_retrieve(self):
        store = build_vector_store(self.documents, self.embeddings)
        with TemporaryDirectory() as temporary_dir:
            index_dir = Path(temporary_dir) / "faiss_index"
            save_vector_store(store, index_dir)

            self.assertTrue(index_exists(index_dir))
            self.assertGreater((index_dir / "index.faiss").stat().st_size, 0)
            self.assertGreater((index_dir / "index.pkl").stat().st_size, 0)

            loaded = load_vector_store(index_dir, self.embeddings)
            self.assertEqual(3, loaded.index.ntotal)
            self.assertEqual(3, loaded.index.d)

            result = retrieve_documents(
                "Como os vetores são armazenados?",
                k=1,
                vector_store=loaded,
            )
            self.assertEqual("faiss.txt", result[0].metadata["filename"])

    def test_retriever_validates_query_and_k(self):
        store = build_vector_store(self.documents, self.embeddings)
        for invalid_k in (0, -1, True, 1.5):
            with self.subTest(k=invalid_k):
                with self.assertRaises((TypeError, ValueError)):
                    get_retriever(k=invalid_k, vector_store=store)

        for query in ("", "   ", None):
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    retrieve_documents(query, vector_store=store)

    def test_missing_index_is_rejected(self):
        with TemporaryDirectory() as temporary_dir:
            with self.assertRaises(FileNotFoundError):
                load_vector_store(temporary_dir, self.embeddings)


if __name__ == "__main__":
    unittest.main()

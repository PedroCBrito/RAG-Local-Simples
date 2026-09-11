import unittest

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.rag_chain import (
    NOT_FOUND_RESPONSE,
    extract_sources,
    format_documents,
    get_rag_chain,
    normalize_query,
)


class RagChainTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="O projeto utiliza Python.",
                metadata={"filename": "manual.pdf", "page": 1, "chunk_index": 2},
            )
        ]

    def test_chain_returns_answer_documents_and_sources(self):
        retriever = RunnableLambda(lambda _query: self.documents)
        llm = FakeListChatModel(responses=["A linguagem é Python."])
        chain = get_rag_chain(retriever=retriever, llm=llm)

        result = chain.invoke("Qual linguagem é utilizada?")

        self.assertEqual("A linguagem é Python.", result["answer"])
        self.assertEqual(self.documents, result["source_documents"])
        self.assertEqual(
            [{"position": 1, "filename": "manual.pdf", "page": 2, "chunk_index": 2}],
            result["sources"],
        )

    def test_context_format_and_empty_fallback_contract(self):
        formatted = format_documents(self.documents)
        self.assertIn("fonte=manual.pdf", formatted)
        self.assertIn("página=2", formatted)
        self.assertIn("O projeto utiliza Python.", formatted)
        self.assertEqual("[CONTEXTO VAZIO]", format_documents([]))
        self.assertEqual([], extract_sources([]))
        self.assertIn("contexto fornecido", NOT_FOUND_RESPONSE)

    def test_query_validation(self):
        self.assertEqual("pergunta", normalize_query("  pergunta  "))
        for query in ("", "   ", None):
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    normalize_query(query)


if __name__ == "__main__":
    unittest.main()

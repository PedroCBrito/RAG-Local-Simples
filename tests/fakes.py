"""Doubles determinísticos compartilhados pelos testes."""

from langchain_core.embeddings import Embeddings


class KeywordEmbeddings(Embeddings):
    """Embedding pequeno que aproxima textos por palavras-chave."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            1.0 if "python" in normalized else 0.0,
            1.0 if "faiss" in normalized or "vetor" in normalized else 0.0,
            1.0 if "café" in normalized or "cafe" in normalized else 0.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

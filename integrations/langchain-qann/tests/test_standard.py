"""LangChain's standard vector store tests (langchain-tests)."""
import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_qann import QANNVectorStore


class TestQANNVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> VectorStore:
        return QANNVectorStore(self.get_embeddings())

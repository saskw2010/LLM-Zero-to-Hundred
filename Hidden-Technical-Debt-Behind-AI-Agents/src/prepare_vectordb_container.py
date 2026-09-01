import os
from typing import List

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from utils.load_config import LoadConfig

load_dotenv()


class PrepareVectorDB:
    def __init__(
        self,
        data_directory: str,
        collection_name: str,
        embedding_model_engine: str,
        chunk_size: int,
        chunk_overlap: int,
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self.data_directory = data_directory
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embedding = OpenAIEmbeddings(model=embedding_model_engine)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _load_all_documents(self) -> List:
        paths = (
            self.data_directory
            if isinstance(self.data_directory, list)
            else [
                os.path.join(self.data_directory, name)
                for name in os.listdir(self.data_directory)
            ]
        )
        docs = []
        for path in paths:
            docs.extend(PyPDFLoader(path).load())
        return docs

    def prepare_and_save_vectordb(self):
        documents = self.text_splitter.split_documents(self._load_all_documents())
        store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding,
            url=self.qdrant_url,
            collection_name=self.collection_name,
            force_recreate=True,
        )
        print(f"Qdrant collection {self.collection_name!r} created.")
        return store


if __name__ == "__main__":
    CFG = LoadConfig(load_vector_store=False)
    PrepareVectorDB(
        data_directory=CFG.data_directory,
        collection_name=CFG.collection_name,
        embedding_model_engine=CFG.embedding_model,
        chunk_size=CFG.chunk_size,
        chunk_overlap=CFG.chunk_overlap,
        qdrant_url=os.getenv("QDRANT_URL_CONTAINER", "http://qdrant:6333"),
    ).prepare_and_save_vectordb()

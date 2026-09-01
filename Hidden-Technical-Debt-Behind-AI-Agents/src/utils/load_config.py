import os
from pyprojroot import here
from yaml import load, Loader
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


class LoadConfig:
    def __init__(self, load_vector_store: bool = True):
        with open(here("config/config.yml"), "r") as f:
            config = load(f, Loader=Loader)

        self.data_directory = str(here(config["directories"]["data_directory"]))
        self.stored_vectordb_dir = str(here(config["directories"]["vectordb_dir"]))

        self.rag_llm = init_chat_model(
            config["model_config"]["rag_model"],
            model_provider=config["model_config"]["model_provider"],
        )
        self.chat_llm = init_chat_model(
            config["model_config"]["chat_llm"],
            model_provider=config["model_config"]["model_provider"],
        )
        self.chat_llm_system_message = config["model_config"]["chat_llm_system_message"]
        self.temperature = config["model_config"]["temperature"]
        self.embedding_model = config["model_config"]["embedding_model"]
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        self.collection_name = config["rag_config"]["collection_name"]
        self.k = config["rag_config"]["k"]
        self.chunk_size = config["rag_config"]["chunk_size"]
        self.chunk_overlap = config["rag_config"]["chunk_overlap"]

        os.environ["LANGSMITH_TRACING"] = "true"

        self.setting = config["setting"]
        if self.setting == "local":
            qdrant_url = os.getenv("QDRANT_URL_LOCAL", "http://localhost:6333")
            self.db_uri = os.getenv("DATABASE_URI_LOCAL")
        elif self.setting == "container":
            qdrant_url = os.getenv("QDRANT_URL_CONTAINER", "http://qdrant:6333")
            self.db_uri = os.getenv("DATABASE_URI_CONTAINER")
        else:
            raise ValueError(f"Unsupported setting: {self.setting!r}")

        if load_vector_store:
            self.stored_vectordb = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=qdrant_url,
            )

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="credit_collection",
    embedding_function=embeddings,
    persist_directory="./credit_db",  # Where to save data locally, remove if not necessary
    collection_metadata={"hnsw:space": "cosine"}
)

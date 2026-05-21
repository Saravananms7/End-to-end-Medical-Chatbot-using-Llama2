import os
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "quickstart")


def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the environment or .env file")
    return Pinecone(api_key=PINECONE_API_KEY)


def _patch_query_for_langchain(index):
    """LangChain 0.0.x calls index.query([vector], top_k=...) positionally."""
    original_query = index.query

    def query(*args, **kwargs):
        if args:
            vector = args[0]
            if (
                isinstance(vector, list)
                and vector
                and isinstance(vector[0], (list, tuple))
            ):
                vector = vector[0]
            kwargs["vector"] = vector
        return original_query(**kwargs)

    index.query = query
    return index


def get_index(index_name: Optional[str] = None):
    pc = get_pinecone_client()
    name = index_name or PINECONE_INDEX_NAME
    return _patch_query_for_langchain(pc.Index(name))

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
from src.pinecone_client import get_index

load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone and upload embeddings to the index
index = get_index()
docsearch = PineconeVectorStore(index, embeddings.embed_query, "text")
docsearch.add_texts([t.page_content for t in text_chunks])

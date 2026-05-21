from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.pinecone_client import get_index

app = Flask(__name__)

load_dotenv()

embeddings = download_hugging_face_embeddings()

# Initialize Pinecone and connect to the index
index = get_index()
docsearch = PineconeVectorStore(index, embeddings.embed_query, "text")


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens': 128,
                          'temperature': 0.7,
                          'threads': 4})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a message.", 400
    print(f"Query: {msg}")
    try:
        print("Retrieving context and generating answer (may take 1-3 min on CPU)...")
        result = qa({"query": msg})
        answer = str(result["result"]).strip()
        print(f"Response: {answer[:200]}...")
        return answer or "I could not generate an answer. Please try again."
    except Exception as e:
        print(f"Error: {e}")
        return f"Sorry, something went wrong: {e}", 500



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False, threaded=True)



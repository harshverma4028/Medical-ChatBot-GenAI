from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
# from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import*
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

# Ensure your Pinecone index exists
index_name = "medicalbot"

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

host = "https://medicalbot-ckos228.svc.aped-4627-b74a.pinecone.io"

# Create the index instance with both index name and host
index = pc.Index("medicalbot", host=host)

# Embed each chunk and upsert the embedding inot 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up the retriever for similarity search
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize OpenAI model
llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Use the new method for combining documents and creating a retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever ,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response : ", response)
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

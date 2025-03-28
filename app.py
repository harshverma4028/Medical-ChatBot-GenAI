from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore 
from langchain_anthropic import ChatAnthropic 
from langchain.chains import RetrievalQA  
from langchain.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
import os
import pinecone

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download embeddings (HuggingFace Embeddings remains unchanged)
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone using the correct method
pinecone.init(api_key=PINECONE_API_KEY)

# Ensure your Pinecone index exists
index_name = "medicalbot"
if index_name not in pinecone.list_indexes():
    print(f"Index '{index_name}' not found. Creating a new one.")
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on the dimensionality of your embeddings
        metric='euclidean'  # Or 'cosine', 'dotproduct', depending on your choice
    )

# Create Pinecone VectorStore using the existing index
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
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Assuming 'stuff' is the chain type you need for your task
    retriever=retriever
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = qa_chain.run(input)  # Use the newer run method for chain execution
    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

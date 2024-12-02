from openai import OpenAI
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from pathlib import Path
from langchain.schema import Document
import streamlit as st

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

SUPPORTED_EXTENSIONS = {'.py', '.md', '.txt', '.js', '.html', '.css', '.jsx', '.ipynb', '.java',
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        print(f"Error reading repository: {str(e)}")
    return files_content

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
pinecone_index = pc.Index("codebase-rag")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ['GROQ_API_KEY']
)

def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)
    
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(), 
        top_k=2, 
        include_metadata=True, 
        namespace="https://github.com/CoderAgent/SecureAgent"
    )
    
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n" + query
    
    system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.
    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """
    
    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    
    return llm_response.choices[0].message.content

# Streamlit UI
st.title("Codebase Q&A")
query = st.text_input("Ask a question about the codebase:")

if query:
    response = perform_rag(query)
    st.write(response)
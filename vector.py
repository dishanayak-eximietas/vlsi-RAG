import streamlit as st
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Streamlit UI
st.title("VLSI-RAG: Document Upload and Chunking")

# Upload OpenAI API Key
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Upload document
uploaded_files = st.file_uploader("Upload documents (PDF, Excel, or TXT)", accept_multiple_files=True)

# Select chunk size and chunk overlap
chunk_size = st.slider("Select chunk size", min_value=100, max_value=2000, value=1000, step=100)
chunk_overlap = st.slider("Select chunk overlap", min_value=0, max_value=500, value=0, step=10)

def load_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path, extract_images=False)
        elif uploaded_file.name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error("Unsupported file type")
            return []

        all_docs.extend(loader.load_and_split())
        os.remove(file_path)
    return all_docs

if uploaded_files:
    st.write("Loading and chunking documents...")
    pages = load_documents(uploaded_files)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False)
    doc_chunks = text_splitter.create_documents([doc.page_content for doc in pages])
    st.success(f"Split into {len(doc_chunks)} chunks.")

    # Vector embedding
    model_name = "BAAI/bge-small-en"
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

    # Create a unique database for each document
    for i, uploaded_file in enumerate(uploaded_files):
        db = FAISS.from_documents(doc_chunks, hf)
        db_name = f"faiss_{uploaded_file.name.replace('.', '_')}"
        db.save_local(db_name)
        st.success(f"Database '{db_name}' created for document '{uploaded_file.name}'.")

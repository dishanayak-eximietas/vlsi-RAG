### Streamlit UI.
### Prints only context, answer and citations




import streamlit as st
import openai
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

# Streamlit UI
st.title("VLSI-RAG")

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

    db = FAISS.from_documents(doc_chunks, hf)
    db.save_local("faiss")
    new_db = FAISS.load_local("faiss", hf, allow_dangerous_deserialization=True)
    st.success("Database created for document retrieval.")

    # Query input
    question = st.text_input("Enter your query:")
    if question:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=new_db.as_retriever(), llm=llm)
        unique_docs = retriever_from_llm.invoke(question)

        template = """
        You are an expert VLSI Design verification engineer answering VLSI domain-related questions.
        Context: {context}
        Question: {question}
        Provide an answer using JSON format with keys 'context', 'answer', 'citations'.
        Assistant:"""

        class VLSI_Parser(BaseModel):
            context: str = Field(description="Boolean indicating if context is present")
            citations: str = Field(description="List of citation IDs")
            answer: str = Field(description="LLM-generated answer")

        parser = JsonOutputParser(pydantic_object=VLSI_Parser)
        prompt = ChatPromptTemplate.from_template(template=template, partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt | llm

        st.write("Generating response...")
        llm_resp = chain.invoke({"question": question, "context": unique_docs})

        # Extract content from AIMessage
        llm_resp_content = llm_resp.content

        # Debug: Print raw response content
        st.write("Raw LLM Response Content:")
        st.write(llm_resp_content)

        # Parse the LLM response
        try:
            response_json = json.loads(llm_resp_content)
            context = response_json.get("context", "")
            answer = response_json.get("answer", "")
            citations = response_json.get("citations", "")

            st.subheader("Response:")
            st.write(f"**Context:** {context}")
            st.write(f"**Answer:** {answer}")
            st.write(f"**Citations:** {citations}")
        except json.JSONDecodeError:
            # st.error("Failed to parse the LLM response.")
            pass

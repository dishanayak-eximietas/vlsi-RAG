import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import json
from dotenv import load_dotenv

load_dotenv()

# Streamlit UI
st.title("VLSI-RAG: Query and Response")

# Upload OpenAI API Key
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# List available databases
available_dbs = [f for f in os.listdir() if f.startswith("faiss_")]
selected_db = st.selectbox("Select a database to query:", available_dbs)

if selected_db:
    model_name = "BAAI/bge-small-en"
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    new_db = FAISS.load_local(selected_db, hf, allow_dangerous_deserialization=True)
    st.success(f"Loaded database '{selected_db}' for querying.")

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

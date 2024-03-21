import logging
import os
import sys
import time

import snowflake.connector
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_snowpoc.embedding import SnowflakeEmbeddings
from langchain_snowpoc.llms import Cortex
from langchain_snowpoc.vectorstores import SnowflakeVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["V_CONNECTION_NAME"] = "cortex_user_183"

MODEL_LLM = "mistral-7b"
MODEL_EMBEDDINGS = "e5-base-v2"
VECTOR_LENGTH = 786


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        connection_name=os.environ.get("V_CONNECTION_NAME"),
    )


snowflake_connection = get_connection()

if "vector" not in st.session_state:

    st.session_state.embeddings = SnowflakeEmbeddings(
        connection=snowflake_connection, model=MODEL_EMBEDDINGS
    )

    st.session_state.loader = WebBaseLoader("https://paulgraham.com/greatwork.html")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vector = SnowflakeVectorStore.from_documents(
        st.session_state.documents,
        st.session_state.embeddings,
        vector_length=VECTOR_LENGTH,
    )

st.title("Chat with Docs - Snowflake Edition :) ")

llm = Cortex(connection=snowflake_connection, model=MODEL_LLM)

prompt = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
# from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader = TextLoader("sample.txt")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=20,chunk_overlap=9)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the given context.  
    Please provide the most accurate response based on the context 
    <context>
    {context}
    <context>
    Questions:{input}
    """
    
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriver, document_chain)
prompt = st.text_input("Input your prompt here")

if prompt:
    response=retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])
    
    with st.expander("Document Similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------")
        
        
    

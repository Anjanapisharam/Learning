from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assistant. Please answer to the User Query."), 
        ("user", "Question:{question}")
    ]
    )



## Streamlit Framework

st.title('Langchain Demo with Ollama')
input_text = st.text_input("Search the Topic here")


## ollama LLM

llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
   
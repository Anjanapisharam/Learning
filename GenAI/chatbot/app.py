from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

os.environ["OPENAI_API_KEY"]=os.getenv("OPEN_AI_TOKEN")
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

st.title('Langchain Demo with OPENAI API')
input_text = st.text_input("Search the Topic here")

## OpenAI LLM

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
    
# run the code :  python -m streamlit run app.py. This will open the interaction window.


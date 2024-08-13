from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

os.environ["OPENAI_API_KEY"]=os.getenv("OPEN_AI_TOKEN")

app = FastAPI(
    title = "Lanchain Server",
    version = "1.0",
    description = "A simple API Server"
)

add_routes(
    app, 
    ChatOpenAI(), 
    path="/openai")

# model=ChatOpenAI()
# llm_1 = ChatOpenAI(" ") #mention the model name if you want to use OpenAI model
# llm_1 = Ollama("gpt-3.5-turbo") 

# llm_2 = Ollama(model="llama2")
llm = Ollama(model="llama2")


prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic} with 100 words")

add_routes( 
    app,
    prompt1|llm,
    path = "/essay"
)

add_routes(
    app,
    prompt2|llm,
    path = "/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    
    
#run command : python app.py
#Go to http://localhost:8000/docs


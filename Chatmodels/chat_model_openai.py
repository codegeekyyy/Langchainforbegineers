from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv
import os

load_dotenv()
# Optional: print to check if key is loaded
# print(os.getenv("OPENAI_API_KEY"))
model=ChatOpenAI(model='gpt-4.1')
res=model.invoke("what is the capital of India?")
print(res)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation")
model=ChatHuggingFace(llm=llm)
res=model.invoke("who is the current president of USA?")
print(res.content)
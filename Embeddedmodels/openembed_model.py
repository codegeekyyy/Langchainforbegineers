from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=["Delhi is the capital of India",
           "Kolkata is the capital of West Bengal",
           "Mumbai is the capital of Maharashtra"]

vector=embedding.embed_documents(documents)
print(vector)
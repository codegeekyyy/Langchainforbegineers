from langchain_community.retrievers import WikipediaRetriever
from langchain_ollama import OllamaLLM

model=OllamaLLM(model="gemma3:4b")

# Initialize the retriever (optional: set language and top_k)
retrive=WikipediaRetriever(top_k_results=2, lang="en")

# define your query 
query="the geopolitical history of india and pakistan from the perspective of a chinese"

docs=retrive.invoke(query)

print(docs)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# source document
documents=[
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# intialize embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# create chroma vector store in memory
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_collection"
)

# convert vector store in retriever
retriever=vectorstore.as_retriever(search_kwargs={"k":2})

query="what is chroma used for?"
res=retriever.invoke(query)

for i, doc in enumerate(res):
    print(f"\n--Result{i+1}--")
    print(doc.page_content)
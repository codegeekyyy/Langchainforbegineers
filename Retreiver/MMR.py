from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

docs=[
     Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# intialize embedding
embeddings=OllamaEmbeddings(model="nomic-embed-text")

# create a FAISS vector store from documents
vectorstore=FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# enable MMR in the retriver
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3,"lambda_mult":1} #lambda_mult-tells relevance-diversity balance
)

query="what is langchain?"
res=retriever.invoke(query)

for i,doc in enumerate(res):
    print(f"\n--Result{i+1}--")
    print(doc.page_content)
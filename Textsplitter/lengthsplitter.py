from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"C:\Users\harsh\OneDrive\Desktop\Langchain-Models\dl-curriculum.pdf")
docs=loader.load()


splitter=CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator=''
)


res=splitter.split_documents(docs)
print(res[1].page_content)

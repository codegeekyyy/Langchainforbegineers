from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


loader=PyPDFLoader(r"C:\Users\harsh\OneDrive\Desktop\Langchain-Models\dl-curriculum.pdf")
loader.load()

model=OllamaLLM(model="gemma3:4b")

prompt=PromptTemplate(
    template="Write a summary of the following pdf content:\n\n{content}",
    input_variables=["content"]
)

parser=StrOutputParser()

chain=prompt|model|parser

print(chain.invoke({"content":loader.load()[0].page_content}))
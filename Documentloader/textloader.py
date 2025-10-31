from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



model=OllamaLLM(model="gemma3:4b")

prompt=PromptTemplate(
    template="Write a summary for the following text:\n\n{text}",
    input_variables=["text"]
)

parser=StrOutputParser()

loader = TextLoader(r"C:\Users\harsh\OneDrive\Desktop\Langchain-Models\cricket.txt", encoding="utf-8")
docs = loader.load()

# print(docs)
chain=prompt|model|parser
print(chain.invoke({"text":docs[0].page_content}))

# print(docs[0].page_content)
# print(docs[0].metadata)
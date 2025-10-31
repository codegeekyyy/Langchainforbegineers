import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyApp/1.0)"

from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = OllamaLLM(model="gemma3:4b")

url = "https://www.flipkart.com/msi-thin-15-intel-core-i7-13th-gen-13620h-16-gb-512-gb-ssd-windows-11-home-6-graphics-nvidia-geforce-rtx-4050-144-hz-45-w-b13ve-gaming-laptop/p/itm8084b0b7961d4?pid=COMHCQ4TFPZPJKMY"
loader = WebBaseLoader(url)
docs = loader.load()

question=input("Enter your question about the webpage: ")

prompt = PromptTemplate(
    template=(
        "Answer the following question based on the provided webpage text:\n\n"
        "Question: {question}\n\n"
        "Webpage Content:\n{text}"
    ),
    input_variables=["question", "text"]
)

parser = StrOutputParser()
chain = prompt | model | parser

response = chain.invoke({
    "question": "What is the price of the product?",
    "text": docs[0].page_content
})

print("Answer:", response)

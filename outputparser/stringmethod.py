from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
 # ✅ FIXED import

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Prompt 1: Generate a detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}.',
    input_variables=['topic']
)

# Prompt 2: Generate a 5-line summary of the report
template2 = PromptTemplate(
    template='Write a 5-line summary of the following text:\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()  # ✅ use StrOutputParser instead of StringOutputParser

# Create a chain
chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({'topic': 'Artificial Intelligence'})

print(res)

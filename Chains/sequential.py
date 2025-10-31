from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
 

 
model=Ollama(model="mistral")

# ✅ Prompts
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following report into 5 key points:\n{report}",
    input_variables=["report"]
)

parser = StrOutputParser()

report_chain = prompt1 | model | parser

# Step 2: Summarize report
summary_chain = prompt2 | model | parser

# Run step by step
report = report_chain.invoke({"topic": "Artificial Intelligence"})
summary = summary_chain.invoke({"report": report})
print("Detailed Report:\n", report)
print("Final Summary:\n", summary)# Chains/sequential.py

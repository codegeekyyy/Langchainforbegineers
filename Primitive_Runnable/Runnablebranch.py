# report_chain.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

prompt_1 = PromptTemplate(
    template="Write a detailed report on the topic: {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=['text']
)

model = OllamaLLM(model="mistral")
parser = StrOutputParser()

report_gen_chain = prompt_1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt_2 | model | parser),
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain


def generate_report(topic: str):
    """Generate report and possibly summary."""
    return final_chain.invoke({'topic': topic})

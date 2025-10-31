from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


model=OllamaLLM(model="gemma3:4b")

prompt_1=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

prompt_2=PromptTemplate(
    template="{joke}\n\nProvide a short explanation of why this joke is funny.",
    input_variables=['joke']
)

chain=RunnableSequence(prompt_1,model,parser,prompt_2,model,parser)

print(chain.invoke({'topic':'AI assistants'}))  
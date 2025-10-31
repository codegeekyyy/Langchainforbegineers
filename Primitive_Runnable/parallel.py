# RunnableParallel-> is a runnnable that allows multiple runnable to execute in parralel each runnable recevies the same input and process it independently, producing a dictionary of outputs

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

prompt_1=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt_2=PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=['topic']
)

model=OllamaLLM(model="gemma3:4b")

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'Tweet':RunnableSequence(prompt_1,model,parser),
    'Linkedin':RunnableSequence(prompt_2,model,parser)
})

res=parallel_chain.invoke({'topic':'the future of work'})
print(res['Tweet'])
print(res['Linkedin'])
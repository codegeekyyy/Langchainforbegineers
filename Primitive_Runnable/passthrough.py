#Runnable passthrough is a special primitive that simply return the input as output without any modification

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

prompt_1=PromptTemplate(
     template='give me a funny response on this: {topic}',
    input_variables=['topic']
)

prompt_2=PromptTemplate(
    template="give me the explanation of that: {topic}",
    input_variables=['topic']
)

model=OllamaLLM(model="gemma3:4b")

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt_1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt_2,model,parser)
})

final_chain=RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

res=final_chain.invoke({'topic':'Cricket'})
print(res['joke'])
print(res['explanation'])
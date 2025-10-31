from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda

# Step 1: Define a prompt
prompt_1 = PromptTemplate(
    template='Provide a motivational quote about {topic}',
    input_variables=['topic']
)

# Step 2: Model
model = OllamaLLM(model="gemma3:4b")

# Step 3: Parser
parser = StrOutputParser()

# Step 4: Combine prompt → model → parser into a single chain
quote_chain = prompt_1 | model | parser

# Step 5: Parallel chain that does two things:
#  - Passes the quote output directly
#  - Calculates the word count
parallel_chain = RunnableParallel({
    'quote': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

# Step 6: Combine them: run quote_chain first, then parallel processing
final_chain = quote_chain | parallel_chain

# Step 7: Invoke
res = final_chain.invoke({'topic': 'perseverance'})

print(res['quote'])
print(f"Word Count: {res['word_count']}")

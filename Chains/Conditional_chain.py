from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
 
model=OllamaLLM(model="gemma3:4b")

parser= StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="give the sentiment of the feedback")
    
parser_2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="classify the following movie review as Positive or Negative:\n\n{review} \n {format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser_2.get_format_instructions()}
)

classifier_chain=prompt1 | model | parser_2

prompt_2=PromptTemplate(
    template="write an appropriate response to this positive review:\n\n{review}",
    input_variables=["review"]
)

prompt_3=PromptTemplate(
    template="write an appropriate response to this negative review:\n\n{review}",
    input_variables=["review"]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='Positive', prompt_2 | model | parser),
    (lambda x:x.sentiment=='Negative', prompt_3 | model | parser),
    RunnableLambda(lambda : "Invalid sentiment" )
)

chain=classifier_chain | branch_chain
res=chain.invoke({'review':"it was dissapointing and boring movie. I want my money back"})
print(res)
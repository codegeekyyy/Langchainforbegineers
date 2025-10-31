from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class person(BaseModel):
    name: str = Field(description="name of the person"),
    age: int = Field(gt=18, description="age of the person"),

parser=PydanticOutputParser(pydantic_object=person)

template=PromptTemplate(
  template="Generate a random name and age of person of a fictional character with {place}.\n {format_instructions  }",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt=template.format_prompt(place="India")
chain = prompt | model | parser
res=chain.invoke({})
print(res)
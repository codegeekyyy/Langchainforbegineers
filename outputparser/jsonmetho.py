from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser



load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(template='facts about black hole. \n {format_instruction}', input_variables=[], partial_variables={'format_instruction':parser.get_format_instructions()})

# prompt=template.format() 
# res=model.invoke(prompt)
# result = parser.parse(res.content if hasattr(res, "content") else res)
# print(result)

# using chain insted of running separately which make it simple and create a pipeline
chain=template | model | parser
res=chain.invoke({})
print(res)



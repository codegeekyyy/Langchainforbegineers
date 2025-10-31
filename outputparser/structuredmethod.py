from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema



load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", task="text-generation"
)

model=ChatHuggingFace(llm=llm)


schema=[
    ResponseSchema(name="fact_1", description="fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="fact 3 about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(template='give 3 facts about the {topic} \n {format_instruction}', input_variables=['topic'], partial_variables={'format_instruction':parser.get_format_instructions()})


chain =template | model | parser
prompt=template.invoke({'topic':'black hole'})

res=chain.invoke(prompt)
print(res)

# parser.parse(res.content if hasattr(res, "content") else res)
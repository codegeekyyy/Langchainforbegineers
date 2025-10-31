
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

 # ✅ FIXED import

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template="tell me a story about a {animal} who loves {food}"
prompt=PromptTemplate.from_template(template)

final_prompt=prompt.format(animal="dog", food="bones")
response=model.invoke(final_prompt)
print(response)
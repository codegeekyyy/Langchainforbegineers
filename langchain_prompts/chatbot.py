
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

 # ✅ FIXED import

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_template=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("user", "English -> {text}"),
])

final_prompt=chat_template.format_messages(text="I love You.")
res=model.invoke(final_prompt)
print(res.content )
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

model=ChatAnthropic(model='claude-3-5-sonnet-20241022')

res=model.invoke("what is the capital of India?")
print(res)
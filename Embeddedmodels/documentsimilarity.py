from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# load_dotenv()

embedding=HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True} 
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me best bowler in India?"
doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding], doc_embeddings)[0]
index, score=sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]# (index,score)
print(query)
print(documents[index])
print("similarity score:", score)


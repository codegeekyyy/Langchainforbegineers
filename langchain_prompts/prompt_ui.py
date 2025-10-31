from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import load_prompt # Assuming 'load_prompt' is correctly imported/defined

load_dotenv() # take environment variables from .env.

# 1. Define the Embedding Model (Used for RAG, not for simple summarization)
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# 2. Define the LLM (Used for Summarization)
# You MUST replace 'your-hf-llm-model' with a valid Hugging Face LLM model name
# and ensure your HF_TOKEN is set in the .env file.
# This assumes you want to use the HuggingFaceEndpoint class.
try:
    model = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Example LLM
        task="text-generation",
        temperature=0.1
    )
except Exception as e:
    st.error(f"Failed to initialize the LLM. Check your model name and HF_TOKEN in .env. Error: {e}")
    st.stop()


st.header("Research tool with LLMs and LangChain")

# Define UI elements ONLY ONCE
paper_input = st.selectbox(
    "Select Research Paper Name",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

# Use only ONE 'if st.button' block and chain the prompt to the LLM (model)
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })

    # The result from a LangChain run or LLM invocation is often an object, 
    # so accessing '.content' is usually correct for the output text.
    st.write(result.content)
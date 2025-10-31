import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import tempfile

# Streamlit UI
st.set_page_config(page_title="PDF Summarizer with Ollama", page_icon="📄", layout="centered")
st.title("📄 PDF Summarizer using Ollama LLM")
st.write("Upload a PDF file, and the model will generate a concise summary for you.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("Processing your PDF... Please wait ⏳")

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Combine all pages into one text
        pdf_text = " ".join(page.page_content for page in pages)

        # Define LLM model
        model = OllamaLLM(model="gemma3:4b")

        # Define prompt
        prompt = PromptTemplate(
            template="Write a detailed and well-structured summary of the following content:\n\n{content}",
            input_variables=["content"]
        )

        parser = StrOutputParser()
        chain = prompt | model | parser

        # Run model
        summary = chain.invoke({"content": pdf_text})

        st.success("✅ Summary generated successfully!")
        st.subheader("📝 PDF Summary:")
        st.write(summary)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("👆 Please upload a PDF file to get started.")

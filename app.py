# app.py
import streamlit as st
from Primitive_Runnable.Runnablebranch import generate_report

st.set_page_config(page_title="AI Report Generator", page_icon="🧠", layout="centered")

st.title("🧠 AI Report Generator using LangChain + Ollama")
st.write("Enter a topic below and let the AI generate a detailed report. If the report is too long, it will also summarize it automatically!")

topic = st.text_input("Enter a topic:", placeholder="e.g., Artificial Intelligence and its impact on modern society")

if st.button("Generate Report"):
    if topic.strip() == "":
        st.warning("Please enter a topic before proceeding.")
    else:
        with st.spinner("Generating your report... Please wait ⏳"):
            try:
                report = generate_report(topic)
                st.success("✅ Report generated successfully!")

                # Display full report
                st.subheader("📘 Full Report:")
                st.write(report)

            except Exception as e:
                st.error(f"An error occurred: {e}")

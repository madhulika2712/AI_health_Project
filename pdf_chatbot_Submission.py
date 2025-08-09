#!/usr/bin/env python
# coding: utf-8

#pip install streamlit pymupdf openai

import streamlit as st
import fitz
import re
import openai

# Azure OpenAI credentials
openai.api_type = "azure"
openai.api_base = "your_azure_openai_base_url"  # Replace with your actual base URL
openai.api_version = "your_api_version"  # Replace with your actual API version
openai.api_key = "your_api_key_here"  # Replace with your actual API key

# Define the section headings you want to extract
target_sections = [
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "DRUG INTERACTIONS"
]

# Function to extract target sections from PDF
def extract_sections_from_pdf(pdf_file):
    pdf_file.seek(0)  # Reset the pointer to the start
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = doc[0].get_text()  # Only using first page
    section_pattern = r"-{5,}(.*?)\-{5,}"
    matches = re.split(section_pattern, text)

    structured = {}
    for i in range(1, len(matches), 2):
        heading = matches[i].strip().upper()
        content = matches[i+1].strip()
        if heading in target_sections:
            if heading in structured:
                structured[heading] += "\n\n" + content
            else:
                structured[heading] = content
    return structured

# Function to check for drug name mismatches before asking a question
# Uses spaCy to extract drug names from the question and document text
import spacy
nlp = spacy.load("en_core_sci_sm")  # or use en_core_web_sm

def check_drug_mismatch(question, document_text):
    question_ents = {ent.text.lower() for ent in nlp(question).ents if ent.label_ in {"DRUG", "PRODUCT"}}
    document_ents = {ent.text.lower() for ent in nlp(document_text).ents if ent.label_ in {"DRUG", "PRODUCT"}}
    
    unmatched = question_ents - document_ents
    return unmatched  # Set of unmatched drug names


# Function to generate response using OpenAI
def ask_question(question, context_dict):
    context_text = "\n\n".join([f"{k}:\n{v}" for k, v in context_dict.items()])
    messages = [
        {"role": "system", "content": (
            "You are a helpful medical assistant. ONLY answer using provided document."
            "If the answer is not in the document, respond with: 'The document does not contain information you asked for.'"
            )},
        {"role": "user", "content": f"Here is the document text:\n{context_text}\n\nQuestion: {question}"}
    ]
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messages,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.set_page_config(page_title="FDA Drug Chatbot", layout="centered")

st.title("💊 FDA Drug Chatbot")
st.write("Upload a label document, and ask questions about it.")

uploaded_files = st.file_uploader("📄 Upload PDF", type="pdf", accept_multiple_files=True)

# Store combined extracted data
combined_extracted = {}

if uploaded_files:
    with st.spinner("Analyzing document..."):
        for file in uploaded_files:
            file_extracted = extract_sections_from_pdf(file)
            for key, value in file_extracted.items():
                if key in combined_extracted:
                    combined_extracted[key] += "\n\n" + value
                else:
                    combined_extracted[key] = value
        
    if combined_extracted:
        st.success("Analyzing completed. You can now ask questions.")
        question = st.text_input("Ask a question about the drug:")
        if question:
            with st.spinner("Thinking..."):
                answer = ask_question(question, combined_extracted)
            st.markdown(f"**💬 Answer:** {answer}")
    else:
        st.error("Could not extract target sections from the uploaded PDF.")


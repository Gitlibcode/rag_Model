import os
import requests
import streamlit as st
from pypdf import PdfReader
from docx import Document
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import tempfile

# --- UI Setup ---
st.title("RAG App with Z.AI GLM-4.6")
api_key = st.secrets["ZAI_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("🔑 Enter your OpenAI API Key", type="password")


uploaded_file = st.file_uploader("📁 Upload a document (PDF, DOCX, XLSX)", type=["pdf", "docx", "xlsx"])
user_query = st.text_area("💬 Enter your prompt")

# --- File Readers ---
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages])

def read_word(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def extract_text(file, filename):
    if filename.endswith(".pdf"):
        return read_pdf(file)
    elif filename.endswith(".docx"):
        return read_word(file)
    elif filename.endswith(".xlsx"):
        return read_excel(file)
    return ""

# --- Text Splitting ---
def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

# --- Embedding ---
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks)

# --- Vector Store ---
def create_vector_db(embeddings):
    dim = embeddings.shape[1]
    index = IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Retrieval ---
def retrieve_chunks(query, vector_db, chunks, top_k=3):
    query_embedding = generate_embeddings([query])
    distances, indices = vector_db.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
Answer in one clear sentence with figure if data involved in table and then details description below:"""

# --- LLM Generation ---
def generate_response(api_key, query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = f"""Based on the following context, answer the query:

Context:
{context}

Query: {query}

instruction = """Answer in one clear sentence with figure if data involved in table and then details description below."""


    url = "https://api.z.ai/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "glm-4.6",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        message = result["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or "No answer returned."
    except Exception as e:
        return f"Error: {e}"


# --- Main Logic ---
if st.button("🚀 Run RAG"):
    if not api_key:
        st.error("Please enter your API key.")
    elif not uploaded_file:
        st.error("Please upload a document.")
    elif not user_query.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            text = extract_text(tmp_path, uploaded_file.name)
            chunks = split_text(text)
            embeddings = generate_embeddings(chunks)
            vector_db = create_vector_db(embeddings)
            relevant = retrieve_chunks(user_query, vector_db, chunks)
            response = generate_response(api_key, user_query, relevant)

        #with st.spinner("Generating descriptive summary with OpenAI..."):
           # expanded_summary = expand_with_openai(openai_api_key, response)

        st.success("✅ Response generated!")

        # Display Z.AI output
        st.markdown("### 🧠 Z.AI Response")
        st.text_area("Concise Answer", value=response, height=150, key="zai_output")

        # Display OpenAI expansion
       # st.markdown("### 📄 OpenAI Expansion")
      #  st.text_area("Descriptive Summary", value=expanded_summary, height=200, key="openai_output")

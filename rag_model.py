import streamlit as st
import pypdf
import io
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- Helper Functions ---
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def call_zhipu_api(prompt, api_key):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "glm-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def rag_qa(query, index, text_chunks, embedding_model, api_key, top_k=3):
    query_embedding = embedding_model.encode([query])
    distance, I = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [text_chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return call_zhipu_api(prompt, api_key)

# --- Streamlit UI ---
st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📘 RAG Question Answering with GLM-4.6 (ZhipuAI API)")

api_key = st.text_input("🔑 Enter your ZhipuAI API key:", type="password")
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file and api_key:
    st.success("PDF uploaded and API key received!")
    extracted_text = ""
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    for page in reader.pages:
        extracted_text += page.extract_text()

    text_chunks = split_text_into_chunks(extracted_text)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))

    query = st.text_input("❓ Enter your question:")
    if query:
        with st.spinner("Generating answer..."):
            answer = rag_qa(query, index, text_chunks, embedding_model, api_key)
        st.markdown("### 🧠 Answer")
        st.write(answer)

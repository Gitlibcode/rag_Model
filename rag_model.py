import streamlit as st
import pypdf
import io
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

# --- Helper Functions ---
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def rag_qa(query, index, text_chunks, model, tokenizer, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    distance, I = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [text_chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = answer.find("Answer:")
    if answer_start != -1:
        answer = answer[answer_start + len("Answer:"):].strip()
    return answer

# --- Streamlit UI ---
st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📘 RAG Question Answering with GLM-4")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")
    extracted_text = ""
    reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
    for page in reader.pages:
        extracted_text += page.extract_text()

    text_chunks = split_text_into_chunks(extracted_text)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))

    with st.spinner("Loading GLM-4 model..."):
        model_name = "THUDM/glm-4-9b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Generating answer..."):
            answer = rag_qa(query, index, text_chunks, model, tokenizer, embedding_model)
        st.markdown("### 🧠 Answer")
        st.write(answer)


import pypdf
import io
import faiss
import numpy as np
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import sys

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def rag_qa(query, index, text_chunks, model, tokenizer, embedding_model, top_k=3):
    """
    Retrieves relevant chunks and generates an answer using the RAG approach.
    """
    # Create query embedding
    query_embedding = embedding_model.encode([query])

    # Search the vector store
    distance, I = index.search(np.array(query_embedding), top_k)

    # Get the retrieved chunks
    retrieved_chunks = [text_chunks[i] for i in I[0]]

    # Prepare the prompt for the language model
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate the answer using the language model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # Set pad_token_id for generation
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"], # Explicitly pass the attention mask
        max_length=500, # Increased max_length for potentially longer answers
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id # Use the defined pad_token_id
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the answer to remove the prompt
    answer_start = answer.find("Answer:")
    if answer_start != -1:
        answer = answer[answer_start + len("Answer:"):].strip()
    else:
        # If "Answer:" is not found, return the whole generated text
        pass # Or handle as an error/warning

    return answer

if __name__ == '__main__':
    # --- PDF Processing ---
    pdf_file_path = "SAFe_Explained_Ebook_2025.pdf" # Make sure your PDF is in the same directory

    extracted_text = ""
    try:
        with open(pdf_file_path, 'rb') as f:
            reader = pypdf.PdfReader(io.BytesIO(f.read()))
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_text += page.extract_text()
        print("Successfully extracted text from the PDF.")
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file_path}' was not found. Please make sure you have the PDF file in the same directory.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        sys.exit()

    # --- Text Chunking ---
    text_chunks = split_text_into_chunks(extracted_text)
    print(f"Number of chunks: {len(text_chunks)}")

    # --- Embedding Creation ---
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating chunk embeddings...")
    chunk_embeddings = embedding_model.encode(text_chunks)
    print(f"Shape of chunk embeddings: {chunk_embeddings.shape}")

    # --- Vector Store Building ---
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))
    print(f"FAISS index created with {index.ntotal} vectors.")

    # --- Load Language Model ---
    print("Loading GPT-2 model and tokenizer...")
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    print("Model loaded.")

    # --- Example Query ---
    query = "What is SAFe?"
    print(f"\nQuery: {query}")
    answer = rag_qa(query, index, text_chunks, model, tokenizer, embedding_model)
    print(f"Answer: {answer}")

    # You can add a loop here to allow multiple queries
    # while True:
    #     user_query = input("\nEnter your question (or type 'quit' to exit): ")
    #     if user_query.lower() == 'quit':
    #         break
    #     answer = rag_qa(user_query, index, text_chunks, model, tokenizer, embedding_model)
    #     print(f"Answer: {answer}")

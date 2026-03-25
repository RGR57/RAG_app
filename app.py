import streamlit as st
from pypdf import PdfReader
import cohere
import numpy as np
import time
import io
from groq import Groq

st.title('RAG Chatbot')
st.write("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
    # Read PDF
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    
    # Split into chunks
    chunks = []
    for i in range(0, len(full_text), 800):
        chunks.append(full_text[i:i+1000])
    
    # Clients
    co = cohere.Client("")
    groq_client = Groq(api_key="")
    
    # Embed only once
    if "embeddings_array" not in st.session_state:
        with st.spinner("Embedding chunks... please wait"):
            chunk_embeddings = []
            for i in range(0, len(chunks), 5):
                batch = chunks[i:i+5]
                batch_embeddings = co.embed(
                    texts=batch,
                    model="embed-english-v3.0",
                    input_type="search_document"
                ).embeddings
                chunk_embeddings.extend(batch_embeddings)
                time.sleep(1)
            
            st.session_state.embeddings_array = np.array(chunk_embeddings)
            st.session_state.chunks = chunks
        st.success("Ready! Ask me anything about your PDF.")
    
    embeddings_array = st.session_state.embeddings_array
    chunks = st.session_state.chunks
    
    # Search function
    def search_chunks(query, top_k=3):
        query_embedding = np.array(co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0])
        scores = np.dot(embeddings_array, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [chunks[i] for i in top_indices]
    
    # Answer function
    def ask_paper(question):
        relevant_chunks = search_chunks(question, top_k=3)
        context = "\n\n".join(relevant_chunks)
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a research assistant. Answer questions using only the context provided."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    
    # Chat interface
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner("Thinking..."):
            answer = ask_paper(question)
        st.write("**Answer:**")
        st.write(answer)
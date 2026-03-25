# RAG Chatbot — PDF Question Answering

Upload any PDF and ask questions about it. The app finds the most relevant sections and answers using AI.

## How it works
- Loads and reads the uploaded PDF
- Splits text into overlapping chunks
- Converts chunks to embeddings using Cohere
- On each question, retrieves the 3 most relevant chunks using cosine similarity
- Sends retrieved chunks to LLaMA 3.3 via Groq to generate an answer

## Tech stack
- Streamlit — web interface
- Cohere — text embeddings
- Groq + LLaMA 3.3 70b — answer generation
- NumPy — similarity search
- PyPDF — PDF reading

## Run locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys in `.streamlit/secrets.toml`
4. Run: `streamlit run app.py`

## Live Demo
[Click here to try it](your_streamlit_url_here)

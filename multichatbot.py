import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# LangChain imports for chunking & embeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Configure API keys for other providers
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ======== PDF Reading ========
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# ======== Chunking Methods ========
def chunk_text(text, method="simple", size=500, overlap=50):
    """
    method: simple | character | recursive | token
    """
    if method == "simple":
        return [text[i:i+size] for i in range(0, len(text), size)]
    elif method == "character":
        splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        return splitter.split_text(text)
    elif method == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        return splitter.split_text(text)
    elif method == "token":
        splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
        return splitter.split_text(text)
    else:
        raise ValueError("Unsupported chunking method")

# ======== Embedding Methods ========
def get_embedder(method):
    if method == "gemini":
        def embed_fn(texts):
            return [
                genai.embed_content(
                    model="models/embedding-001",
                    content=txt,
                    task_type="retrieval_document"
                )["embedding"]
                for txt in texts
            ]
        return embed_fn
    elif method == "cohere":
        embedder = CohereEmbeddings(model="embed-english-light-v3.0")
        return lambda texts: embedder.embed_documents(texts)

    else:
        raise ValueError("Unsupported embedding method")
    '''
    elif method == "openai":
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        return lambda texts: embedder.embed_documents(texts)
    

    elif method == "huggingface":
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token="HUGGINGFACEHUB_API_TOKEN"
            )
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        return lambda texts: embedder.embed_documents(texts) '''

    

# ======== FAISS Index ========
def build_index(chunks, embedder):
    embeddings = np.array(embedder(chunks)).astype("float32")
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

def search(query, k=3):
    query_vector = np.array(st.session_state.embedder([query])).astype("float32")
    D, I = st.session_state.index.search(query_vector, k)
    return [st.session_state.chunks[i] for i in I[0]]

# ======== Q&A ========
def answer_question(query):
    if not st.session_state.chunks or st.session_state.index is None:
        return "❗ Please upload and process a PDF first."
    context = "\n".join(search(query))
    prompt = f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ======== Streamlit UI ========
st.set_page_config(page_title="PDF Q&A Bot - Chunking & Embedding Playground", layout="wide")
st.markdown("<h1 style='text-align: center;'>📘 PDF Q&A BOT </h1>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

# === Left Column: Upload & Process ===
with left_col:
    st.markdown("### 📁 Upload & Process PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    chunk_method = st.selectbox("Select Chunking Method", ["simple", "character", "recursive", "token"])
    chunk_size = st.number_input("Chunk Size", value=500, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", value=50, step=10)

    embed_method = st.selectbox("Select Embedding Method", ["gemini", "openai", "huggingface", "cohere"])

    if st.button("Process PDF"):
        if uploaded_file is not None:
            try:
                text = read_pdf(uploaded_file)
                st.session_state.chunks = chunk_text(text, method=chunk_method, size=chunk_size, overlap=chunk_overlap)
                st.session_state.embedder = get_embedder(embed_method)
                st.session_state.index = build_index(st.session_state.chunks, st.session_state.embedder)
                st.success(f"✅ PDF processed using {chunk_method} chunking & {embed_method} embeddings.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please upload a PDF file.")

# === Right Column: Q&A ===
with right_col:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Ask"):
        if query.strip():
            with st.spinner("Generating answer..."):
                answer = answer_question(query)
                st.text_area("Answer", value=answer, height=200)
        else:
            st.warning("Please enter a question.")

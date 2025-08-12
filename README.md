# PDF-Q-A-CHATBOT  
## Chatbot using RAG with Different Chunking Methods and Embedding Methods  

**📘 PDF Q&A Bot — Chunking & Embedding Playground**  
A Streamlit-powered application that allows you to upload a PDF, chunk its content, generate embeddings using multiple providers (Gemini, OpenAI, Cohere, HuggingFace), store them in a FAISS vector index, and ask natural language questions whose answers are retrieved from the PDF content.  

---

## 🚀 Features  

📂 **Upload any PDF and extract its text**.  

🪓 **Multiple chunking methods**:  
- Simple split  
- Character-based split  
- Recursive character split  
- Token-based split  

🧠 **Multiple embedding backends**:  
- Google Gemini  
- OpenAI *(yet to improve)*  
- HuggingFace Sentence Transformers *(yet to improve)*  
- Cohere  

🔍 **Semantic search** using **FAISS** for fast retrieval.  

---

## 📚 How It Works  

1. **Text Extraction** → PDF text is extracted using `pypdf`.  
2. **Chunking** → Text is split into smaller segments to fit embedding model limits.  
3. **Embedding** → Selected embedding model converts chunks into numerical vectors.  
4. **Indexing** → FAISS stores the vectors for efficient similarity search.  
5. **Retrieval** → On a query, similar chunks are retrieved.  
6. **Answer Generation** → Model uses retrieved chunks as context to answer.  

---

## 📊 Results  

### 🔹 Chunking Methods Comparison  

| Chunking Method | Description | Pros | Cons |
|-----------------|-------------|------|------|
| **Simple** | Splits text into fixed-size chunks. | Fast, easy to implement. | May break sentences or lose context. |
| **Character** | Splits by a set number of characters. | Good for uniform text length. | Can split mid-word or mid-sentence. |
| **Recursive** | Splits respecting paragraphs/sentences first. | Preserves context, better for QA. | Slightly slower. |
| **Token** | Splits based on LLM token count. | Prevents token overflow errors. | Requires tokenizer, more complex. |

---

### 🔹 Embedding Methods Comparison  

| Embedding Provider | Model Used | Strengths | Limitations |
|--------------------|------------|-----------|-------------|
| **Google Gemini** | `models/embedding-001` | High quality, multilingual, good semantic understanding. | Requires Google API key. |
| **OpenAI** | `text-embedding-3-small` | High accuracy, widely supported in LangChain. | Paid API after free tier. |
| **Hugging Face** | `sentence-transformers/all-MiniLM-L6-v2` | Free, runs locally, no API needed. | Slightly slower on large docs. |
| **Cohere** | `embed-english-light-v3.0` | Fast, optimized for English. | Limited multilingual support. |

---

### 🔹 Chunk Size & Chunk Overlap  

- **Chunk Size** → The maximum number of characters or tokens in each chunk.  
  - Example: `chunk_size = 500` → each chunk is ~500 characters/tokens.  
  - **Effect**: Larger chunks = more context per query, but risk hitting model token limits.  

- **Chunk Overlap** → The number of characters/tokens that overlap between chunks.  
  - Example: `chunk_overlap = 50` → 50 characters/tokens repeated between consecutive chunks.  
  - **Effect**: Higher overlap = better context retention between chunks, but more processing cost.  

---

## 📷 Output Screenshot  
<img width="1361" height="608" alt="image" src="https://github.com/user-attachments/assets/b24e13e0-7be4-4759-8979-5ff75dff70a2" />

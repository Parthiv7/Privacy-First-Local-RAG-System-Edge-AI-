# **Project Title**



## **Privacy-First Local RAG System (Edge AI)**





A secure, offline-capable Retrieval-Augmented Generation (RAG) pipeline built to analyze sensitive financial documents on consumer hardware using quantized Small Language Models (SLMs).



##### **Technical Architecture**



* **Orchestration Framework:** LangChain (Python)



* **Inference Engine:** Ollama (running locally)



* **Large Language Model:** Llama 3.2 (1B Parameters) – Chosen for low-latency edge performance.



* **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 – HuggingFace local embeddings.



* **Vector Database:** FAISS (Facebook AI Similarity Search) – For efficient similarity search.



User Interface: Streamlit – Interactive chat interface with citation rendering.



##### **Key Features (The "Why")**



* **100% Data Privacy:** The entire pipeline runs locally. No data is sent to external APIs (OpenAI/Anthropic), making it compliant with strict data security standards.



* **Zero-Cost Infrastructure:** Eliminates cloud compute costs by leveraging local CPU/RAM for inference.



* **Latency Optimization:** optimized for "Edge AI" deployment using a quantized 1B model, ensuring rapid responses even on standard laptops.



* **Hallucination Check:** Implements a retrieval-based citation system, ensuring every answer is grounded in specific document chunks (with page numbers).



##### **Data Flow (The "How")**



* **Ingestion:** PDF documents (e.g., Apple 10-K, Google Environmental Reports) are loaded and split into semantic chunks.



* **Embedding:** Text chunks are converted into dense vector representations using all-MiniLM-L6-v2.



* **Indexing:** Vectors are stored in a local FAISS index for high-speed retrieval.



* **Retrieval:** User queries are converted to vectors; the system performs a similarity search to fetch the top k=6 relevant chunks.



* **Generation:** The relevant context + user query are passed to the Llama 3.2 model to generate a natural language response with source attribution.

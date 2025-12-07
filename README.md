
## **Privacy-First Local RAG System (Edge AI)**
![Status: Working](https://img.shields.io/badge/Status-Working-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)

A secure, offline-capable **RAG (Retrieval-Augmented Generation) pipeline** built to analyze financial and legal documents directly on consumer hardware. This project transforms a paid cloud architecture into an **Edge AI solution** by leveraging open-source, local models.

---
## 📸 Application Demo (Side-by-Side Proof)

<div align="center">
    <img src="assets/Screenshot 2025-12-07 001937.png" width="48%" alt="Successful Query and Summary Output">
    &nbsp; &nbsp;
    <img src="assets/Screenshot 2025-12-07 001803.png" width="48%" alt="Streamlit UI Showing the Sidebar and Initial Query">
</div>

*The image on the left shows the final, summarized response from Llama 3.2. The image on the right shows the architecture (sidebar) and the question being asked.*

## Key Features & Value Proposition

* **100% Data Privacy (Zero Trust):** The entire pipeline—from embedding to inference—runs locally. No data is ever sent to external APIs (OpenAI/Google), making it ideal for sensitive, zero-trust environments.
* **Zero-Cost Infrastructure:** Eliminates recurring cloud compute and API usage fees by leveraging local resources instead of paid services.
* **Edge AI Performance:** Optimized for low-latency retrieval using a streamlined `RetrievalQA` chain and a lightweight, fast **Llama 3.2 (1B)** model.
* **Interactive UI:** User-friendly chat interface built with **Streamlit** for real-time querying and source citation.

---

## 🛠️ Technical Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | Python / **LangChain** | Manages the RAG sequence (retrieval, context stuffing, generation). |
| **Inference Engine (Brain)** | **Ollama** | Hosts and serves the LLM locally via API calls. |
| **Large Language Model (SLM)** | **Llama 3.2 (1B)** | The Small Language Model used for generating responses. |
| **Vector Database (Memory)** | **FAISS** | Provides high-speed similarity search for efficient context retrieval. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Converts PDF chunks into mathematical vectors (embeddings). |
| **Interface** | **Streamlit** | Provides the interactive web application frontend. |

---


### How to Run Locally

This project assumes you have **Python 3.10+** and **Ollama** installed.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the LLM:**
    * Ensure Ollama is running in the background.
    * Download the required model:
        ```bash
        ollama run llama3.2:1b
        ```

4.  **Ingest Documents:**
    * Place your `.pdf` documents in the `raw_pdfs/` folder.
    * Build the vector database (this creates the `faiss_index` folder):
        ```bash
        python ingest.py
        ```

5.  **Launch the Application:**
    ```bash
    streamlit run gui.py
    ```
    Your application will open automatically in your browser at `http://localhost:8501`.


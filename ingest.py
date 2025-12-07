import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- CONFIGURATION ---



def ingest():
    # 1. DEFINE SOURCE DIRECTORY
    pdf_folder_path = "raw pdfs"
    all_documents = []

    print(f"Scanning '{pdf_folder_path}' for PDFs...")

    # 2. LOOP THROUGH ALL FILES
    if not os.path.exists(pdf_folder_path):
        print(f"Error: Folder '{pdf_folder_path}' not found.")
        return

    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            print(f"Loading: {filename}...")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    if not all_documents:
        print("No documents loaded. Exiting.")
        return

    print(f"\nTotal loaded pages: {len(all_documents)}")

    # 3. SPLIT TEXT INTO CHUNKS
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    print(f"Created {len(texts)} chunks.")

    # 4. CREATE EMBEDDINGS & SAVE TO DB
    print("Generating Embeddings (this may take a moment)...")
   
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save the database locally so we don't have to rebuild it every time
    vectorstore.save_local("faiss_index")
    print("Success! Database saved to folder 'faiss_index'.")

if __name__ == "__main__":
    ingest()
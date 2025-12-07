import streamlit as st
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="My Local RAG", page_icon="🤖")
# --- SIDEBAR  ---
with st.sidebar:
    st.title("ℹ️ Project Info")
    st.markdown("This is a **RAG (Retrieval-Augmented Generation)** application built to run **100% locally**.")
    
    st.subheader("🛠️ Tech Stack")
    st.write("• **LLM:** Llama 3.2 (1B)")
    st.write("• **Embeddings:** All-MiniLM-L6-v2")
    st.write("• **Vector DB:** FAISS")
    st.write("• **Framework:** LangChain")
    
    st.divider()
    
    # A button to clear chat helps when testing
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
st.title("🤖 Chat with Your PDFs (Local & Private)")

# --- 2. LOAD RESOURCES (CACHED) ---
# We use @st.cache_resource so we only load the heavy AI models ONCE.
@st.cache_resource
def load_resources():
    print("Loading models...")
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load Vector Store
    # allow_dangerous_deserialization is needed for local pickle files
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Setup the Retriever (k=6 for better accuracy as we discussed)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Setup the Brain (Llama 3.2)
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    # Setup the Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Load the chain
try:
    qa_chain = load_resources()
    st.success("Brain & Memory Loaded! 🧠", icon="✅")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- 3. CHAT HISTORY SETUP ---
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLE USER INPUT ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            # Run the RAG Chain
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            
            # Format sources
            sources = result.get("source_documents", [])
            source_text = "\n\n**Sources:**"
            unique_pages = set()
            for doc in sources:
                page = doc.metadata.get("page", "Unknown")
                source = doc.metadata.get("source", "Unknown")
                unique_pages.add(f"{source} (Page {page})")
            
            for page in unique_pages:
                source_text += f"\n- {page}"
            
            # Combine answer + sources
            full_response = answer + source_text

        # Display Response
        message_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
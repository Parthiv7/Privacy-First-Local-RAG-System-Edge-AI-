import os
import langchain
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- 1. CORE IMPORTS (These were likely missing) ---
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
# --- 2. UNIVERSAL AGENT SELECTOR ---
# This block ensures we use the best available Agent constructor
agent_constructor = None

try:
    # Try the modern 2025 standard
    from langchain.agents import create_tool_calling_agent
    agent_constructor = create_tool_calling_agent
    print(f"Loaded Modern Agent: create_tool_calling_agent (LangChain v{langchain.__version__})")
except ImportError:
    try:
        # Fallback to the 2024 standard
        from langchain.agents import create_openai_functions_agent
        agent_constructor = create_openai_functions_agent
        print(f"Loaded Stable Agent: create_openai_functions_agent (LangChain v{langchain.__version__})")
    except ImportError:
        # Last resort for legacy versions
        from langchain.agents import create_openai_tools_agent
        agent_constructor = create_openai_tools_agent
        print(f"Loaded Legacy Agent: create_openai_tools_agent (LangChain v{langchain.__version__})")

# Load Secrets
load_dotenv()

app = Flask(__name__)

# --- 3. LOAD BRAIN ---
print("Loading vector store...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")# This is where it failed before
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    print("Brain loaded successfully! 🧠")
except Exception as e:
    print(f"Error loading database: {e}")
    print("Make sure you have an OpenAI API Key in .env and ran 'ingest.py'")
    exit(1)

# --- 4. DEFINE TOOL (Manual Wrapper) ---
def search_documents(query):
    """Searches the vector database for relevant PDF chunks."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join([d.page_content for d in docs])

pdf_tool = Tool(
    name="search_documents",
    func=search_documents,
    description="Useful for searching the uploaded PDF documents. Input should be a specific question."
)

tools = [pdf_tool]

# --- 5. RAG CHAIN SETUP (Better for Local Models) ---
print("Initializing RAG Chain...")
# 1. Define the Brain (LLM) again here to be safe
# (Make sure you have: from langchain_community.chat_models import ChatOllama at the top)
llm = ChatOllama(model="llama3.2:1b", temperature=0)
# Create the "Chain" that connects the Brain (LLM) to the Memory (Vector Store)
# checks if 'db' or 'vectorstore' exists from your loading step
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' means it stuffs the found text into the prompt
    retriever=retriever,
    return_source_documents=True
)

# --- 6. RUN THE QUERY ---
query = "Summarize the financial performance of Apple based on the documents."
print(f"\nQuerying: {query}\n")

# Run the chain
response = qa_chain.invoke({"query": query})

print("--- RESULT ---")
print(response["result"])
print("\n--- SOURCES ---")
for source in response["source_documents"]:
    print(f"- Page {source.metadata.get('page', 'Unknown')} of {source.metadata.get('source', 'Unknown')}")

# --- 6. RUN THE QUERY ---
# Define the question you want to ask your PDF
query = "Summarize the financial performance of Apple based on the documents."
print(f"\nQuerying: {query}\n")

# ERROR FIX: Use 'qa_chain' (the new method), NOT 'agent_executor' (the old method)
response = qa_chain.invoke({"query": query})

# Print the final answer
print("--- RESULT ---")
print(response["result"])

# Print the sources to verify it read the file
print("\n--- SOURCES ---")
if "source_documents" in response:
    for source in response["source_documents"]:
        print(f"- Page {source.metadata.get('page', 'Unknown')} of {source.metadata.get('source', 'Unknown')}")
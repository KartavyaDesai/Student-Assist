import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from youtubesearchpython import VideosSearch
from langchain_community.vectorstores import FAISS
import tempfile
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


# Page config
st.set_page_config(page_title="StudentAssist - RAG Learning Assistant", layout="wide")
st.title("📚 StudentAssist - Your Learning Companion")

#Using Hugging Face Embeddings to convert words to vectors.
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar for API keys
with st.sidebar:
    st.header("API Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    # hf_token = st.text_input("Enter HuggingFace Token", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your API keys to continue")
        st.stop()

# File upload
uploaded_file = st.file_uploader("Upload your textbook (PDF)", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load and process document
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(pages)

    # Create vector store
    vectorstore=FAISS.from_documents(documents=chunks,embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize Groq LLM
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="Gemma2-9b-It"
    )

    # Create conversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Query input
    query = st.text_input("Ask a question about your textbook:")
    
    if query:
        with st.spinner("Processing your question..."):
            # Get answer from RAG
            result = qa_chain({"question": query, "chat_history": []})
            
            # Display answer
            st.write("### Answer:")
            st.write(result["answer"])
            
            # Search related YouTube videos
            videos_search = VideosSearch(query, limit=3)
            results = videos_search.result()
            
            # Display videos
            st.write("### Related YouTube Videos:")
            cols = st.columns(3)
            
            for idx, video in enumerate(results["result"]):
                with cols[idx]:
                    st.image(video["thumbnails"][0]["url"])
                    st.write(f"[{video['title']}](https://youtube.com/watch?v={video['id']})")

    # Cleanup temp file
    os.unlink(tmp_path)

else:
    st.info("Please upload a PDF textbook to get started")
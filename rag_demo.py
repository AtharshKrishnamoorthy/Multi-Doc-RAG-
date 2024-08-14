from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import bs4
from dotenv import load_dotenv
import pygame
import os
import time
import streamlit as st
import tempfile

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
api_key = os.getenv("DEEPGRAM_API_KEY")

def create_documents(type):
    documents = None  
    tmp_file_path = None  # Initialize the variable here

    if type == 'Text':
        file = st.sidebar.file_uploader("Upload the Text file", key='Text')
        if file:
            file_name = file.name
            loader = TextLoader(file_name)
            documents = loader.load()
    elif type == 'Web':
        url = st.sidebar.text_input("Enter the Url of the website")
        if url:
            loader = WebBaseLoader(web_path=url)
            documents = loader.load()
    elif type == 'PDF':
        file = st.sidebar.file_uploader("Upload the PDF file", key='PDF')
        if file:
            file_name = file.name
            loader = PyPDFLoader(file_name)
            documents = loader.load()
    elif type == 'CSV':
        file = st.sidebar.file_uploader("Upload the CSV File", key="CSV")
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name

            loader = CSVLoader(file_path=tmp_file_path,
                               csv_args={
                                   'delimiter': ',',
                                   'quotechar': '"'
                               })
            documents = loader.load()

    if documents is None:
        st.warning(f"No documents loaded for {type}. Please upload a file or provide a valid input.")
        return None, type

    return documents, type


def create_vectorstore(Type):
    documents, _ = create_documents(Type)
    if documents is None:
        return None

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    doc = text_splitter.split_documents(documents)
    # Creating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(doc, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def llm_response(input):
    # LLM
    llm = ChatGroq(model="llama-3.1-70b-versatile",
                   temperature=0.9)
    
    # Creating Prompt template
    prompt = ChatPromptTemplate.from_template(""" 
    Answer the question based on the context only.
    Provide it in a structured way. 
    <context>
    {context}
    </context>
    Question: {input}""")

    # Creating stuff document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # Creating Retriever Chain
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({"input": input})
   
    return response['answer']





def main():
    st.title("MultiDoc RAG System")
    st.markdown("Ask a query with the uploaded files and get a response instantly.")
    Type = st.sidebar.selectbox("Select the type of File you want to interact with",
                                ("Text", "Web", "PDF", "CSV"))
    
    with st.spinner("Creating Embeddings and Vector Store..."):

        db = create_vectorstore(Type)
        if db is None:
          st.stop() 

    messages = st.container()


    if prompt := st.chat_input("Ask a question"):
        with st.spinner("Generating response..."):
         with messages:

            st.chat_message("user").write(prompt)

            response_text = llm_response(prompt)
            st.chat_message("assistant").write(response_text)
            # Enable this if u want text 2 speech
            # text_to_speech(api_key,response_text,"output.wav")
            # play_audio("output.wav")


    st.markdown("---")
    st.write("Powered by Vext and Streamlit")

if __name__ == "__main__":
    main()
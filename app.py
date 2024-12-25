import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()
hf_token = os.getenv("HF_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the context doesn't have enough information, state that clearly. 

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = HuggingFaceHub(
        repo_id="EleutherAI/gpt-neo-125M", 
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.run(input_documents=docs, question=user_question)
    return response

def main():
    st.set_page_config(page_title="DocuQBot", page_icon="🤖")
    st.title("AI Conversational Assistant for Document Analysis 🤖")
    uploaded_files = st.sidebar.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    st.sidebar.button('Clear Chat History', on_click=lambda: st.session_state.pop('messages', None))

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs processed and indexed!")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Upload a PDF and ask me anything about it."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
import requests
from io import BytesIO



load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


pdf_files = [
    "ch1-unix-programming.pdf",
    "test-syllabus.pdf"
]

def get_absolute_paths(relative_paths):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_paths = [os.path.join(script_dir, path) for path in relative_paths]
    return absolute_paths




# read all pdf files and return text



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any questions about the course!"}]


def save_chat_history():
    return 0



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response




gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, blue);
    background: linear-gradient(to right, #ffa745, #fe869f,#ef7ac8, #a083ed, #43aeff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">Virtual TA Chatbot </div>
"""



def main():
    st.set_page_config(
        page_title="Virtual TA Chatbot",
        page_icon="📝"
    )

    absolute_pdf_paths = get_absolute_paths(pdf_files)
    raw_text = get_pdf_text(absolute_pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.caption("Welcome to the TA's office hours! The TA can answer questions about the course Intro to Unix Fundamentals.")


    # Sidebar
    with st.sidebar:
        st.title("About the Chatbot")
        st.markdown("This chatbot aims to assist students with course-related queries, provide explanations, offer resources, and facilitate discussions.")

        st.divider()

        st.title("Chat Menu")
        st.markdown("Use the buttons below to clear your chatbot history or save your chatbot history.")
        col1, col2 = st.columns([1,1])
        with col1:
            st.button('Clear History', on_click=clear_chat_history)
        with col2:
            st.button('Save History', on_click=save_chat_history)


        st.divider()

        st.title("Disclaimer")
        st.caption("*Please use this TA chatbot responsibly. Asking for answers to assignments, homework, or exams is strictly prohibited.*")



        
        

    # Main content area for displaying chat messages
    # st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me any questions about the course!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

    


if __name__ == "__main__":
    main()
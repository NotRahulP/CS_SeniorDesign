import os
import streamlit as st
import google.generativeai as genai
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CSS Formatting
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
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
</style>
<div class="gradient-text">Virtual TA Chatbot </div>
"""

# Build conversational chain with question and prompt
def get_conversational_chain():
    prompt_template = """
    You are a teacher's assistant for a course about Java programming. You will be asked questions about Java programming and the course logistics.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    the provided context just say, "I'm sorry, I don't know the answer. Please contact johnblob@utd.edu with your question.", but do not provide an 
    incorrect answer or make up an answer. Make sure your answer is grammatically correct, spelled correctly, and complete sentences. You may respond 
    to greetings with greetings. \n\n

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


# Get user input and search for similar pages in FAISS database
def user_input(user_question, faiss_db):
    docs = faiss_db.similarity_search(user_question)
    print(docs)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True, 
        )
    print(response)
    return response


# Load FAISS database
def load_databases(db_type):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.load_local(f"faiss-index-{db_type}", embeddings, allow_dangerous_deserialization=True) 
    return faiss_db


# Clear chat history and reset chat history file
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! How can I help you?"}]
    f = open("Chat_History.txt", "w")
    f.close()
    write_chat(st.session_state.messages[-1])


# Format chat history text file
def format_chat_history(message):
    chat_string = ""
    role = message["role"]
    content = message["content"]
    chat_string += f"{role.capitalize()}: {content}\n"
    return chat_string


# Record the last chat in the text file
def write_chat(msg):
    f = open("Chat_History.txt", "a")
    f.write(format_chat_history(msg))
    f.close()


# Save chat history in pdf file
def save_chat_history():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', '', 12)

    f = open("Chat_History.txt", "r")
    for x in f: 
        pdf.multi_cell(175,5, txt = x, align = 'L') 

    pdf_content = pdf.output(dest="S").encode("latin-1")  # Generate PDF data
    f.close()
    return pdf_content


def main():
    # Load both syllabus and content databases
    syll_faiss_db = load_databases("test-syllabus")
    textbook_faiss_db = load_databases("java-book")

    # Set webpage tab title and icon
    st.set_page_config(
        page_title="Virtual TA Chatbot",
        page_icon="📝"
    )

    # Display page title and caption
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.caption("Welcome to the TA's office hours! The TA can answer questions about the course Intro to Unix Fundamentals.")


    # Sidebar
    with st.sidebar:
        col1, col2 = st.columns([1,2])
        with col1:
            st.image("https://brand.utdallas.edu/files/utd-bug.jpg", width=150)
        with col2:
            st.markdown("This chatbot aims to assist students with course-related queries, provide explanations, offer resources, and facilitate discussions.")

        st.divider()

        st.title("Chat Menu")
        st.markdown("Use the buttons below to select whether you would like to ask questions about the course syllabus or course content.")
        selected_db = st.radio(
            label = "Knowledge base selection",
            options = ["Syllabus", "Content"],
            index = 0,
            captions = [
                "Ask about course timing, location, office hours, due date, and grade breakdowns.",
                "Ask about any course content from the assigned class textbook."
            ],
            label_visibility = "collapsed"
        )

        st.divider()

        col1, col2 = st.columns([1,1])
        with col1:
            st.button('Clear Chat History', on_click=clear_chat_history)
        with col2:
            st.download_button(label="Download Chat History", data=save_chat_history(), file_name="chat_history.pdf")

        st.divider()

        st.title("Disclaimer")
        st.caption("*Please use this TA chatbot responsibly. Asking for answers to assignments, homework, or exams is strictly prohibited.*")


    # Main content area for displaying chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you?"}]
        write_chat(st.session_state.messages[-1])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me any questions about the course!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        write_chat(st.session_state.messages[-1])
        with st.chat_message("user"):
            st.write(prompt)

        if selected_db == "Content":
            faiss_db = textbook_faiss_db
        else:
            faiss_db = syll_faiss_db
        
        with st.spinner("Thinking..."):
            try:
                response = user_input(prompt, faiss_db)
                full_response = ''
                for item in response['output_text']:
                    full_response += item
            except genai.types.generation_types.StopCandidateException as e:
                full_response = "I'm sorry, I cannot answer that. Please contact johnblob@utd.edu with your question."
                print(e)
                    
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        write_chat(st.session_state.messages[-1])


if __name__ == "__main__":
    main()
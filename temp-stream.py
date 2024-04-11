import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
import base64



st.title("Echo Bot")


def format_chat_history(message):
    chat_string = ""
    role = message["role"]
    content = message["content"]
    chat_string += f"{role.capitalize()}: {content}\n"
    return chat_string
    master_array.append(chat_string)
    


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


def save_chat_history():
    print("ok")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', '', 12)

    f = open("Chat_History.txt", "r")
    for x in f: 
        pdf.cell(50,5, txt = x, ln = 1, align = 'L') 
    #pdf.cell(40, 10, "hello")

    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
    st.markdown(html, unsafe_allow_html=True)

    
 
    
    
    #html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")
    #st.markdown(html, unsafe_allow_html=True)

# def save_chat_history():
#     try:
#         with open(f"./data/history_chat/{file_name}.txt", "w", encoding="utf-8") as file:
#             for m in master_array:
#                 file.write(f"{m}\n")
#         st.success("Chat history saved successfully!")
#     except Exception as e:
#             # Display an error message if there's an issue saving the chat history
#             st.error(f"Error saving chat history: {e}") 



st.button('Save History', on_click=save_chat_history)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    #print(message["role"])
    #print(message["content"])
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    #print(format_chat_history(message))

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    #print(format_chat_history(st.session_state.messages[-1]))
    f = open("Chat_History.txt", "a")
    f.write(format_chat_history(st.session_state.messages[-1]))
    f.close()
    

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    #print(format_chat_history(st.session_state.messages[-1]))
    f = open("Chat_History.txt", "a")
    f.write(format_chat_history(st.session_state.messages[-1]))
    f.close()
    
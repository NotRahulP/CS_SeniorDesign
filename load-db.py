import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_absolute_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, relative_path)
    return absolute_path


# read all pdf files and return text
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# split text into chunks
def process_pdf(pdf_path):
    text = get_pdf_text(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000)
    text_chunks = splitter.split_text(text)
    return text_chunks  # list of strings


# get embeddings for each chunk
def make_vector_store(chunks, filename):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(filename)


def main():
    pdf_files = [
        "test-syllabus.pdf",
        "java-book.pdf",
    ]
    for pdf in pdf_files:
        absolute_pdf_path = get_absolute_path(pdf)
        text_chunks = process_pdf(absolute_pdf_path)
        filename = f"faiss-index-{os.path.splitext(pdf)[0]}"  # Create new filename with prefix
        make_vector_store(text_chunks, filename)
          

if __name__ == "__main__":
    main()
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config("Chat with Multiple PDF")

load_dotenv()

genai.configure(api_key="AIzaSyDK3DT3MtOaGvkaahpgQ7i8WReQJ-UgCn0")

# Initialize chat history in session state if it doesn't exist
if 'current_chat_history' not in st.session_state:
    st.session_state['current_chat_history'] = []

def init_db():
    conn = sqlite3.connect("queries.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      query TEXT NOT NULL,
                      response TEXT NOT NULL,
                      timestamp TEXT NOT NULL)''')  
    conn.commit()
    conn.close()

def store_chat(query, response):
    conn = sqlite3.connect("queries.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO chat_history (query, response, timestamp) VALUES (?, ?, ?)", (query, response, timestamp))
    conn.commit()
    conn.close()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text if page_text else ""

            # Check for annotations, including links (hyperlinks)
            if "/Annots" in page:
                annotations = page["/Annots"]
                for annotation in annotations:
                    annotation_obj = annotation.get_object()
                    if annotation_obj.get("/Subtype") == "/Link":
                        link = annotation_obj.get("/A", {}).get("/URI")
                        if link:
                            text += f"\nHyperlink: {link}\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDK3DT3MtOaGvkaahpgQ7i8WReQJ-UgCn0")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, api_key="AIzaSyDK3DT3MtOaGvkaahpgQ7i8WReQJ-UgCn0")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDK3DT3MtOaGvkaahpgQ7i8WReQJ-UgCn0")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Store chat history in the session state
    st.session_state['current_chat_history'].append({"query": user_question, "response": response["output_text"]})

    # Display the reply
    st.markdown('<span style="color: lightblue;">Reply:</span> ' + response["output_text"], unsafe_allow_html=True)

    # Store query and response in the database
    store_chat(user_question, response["output_text"])


def show_current_chat_history():
    st.write("---")
    for chat in reversed(st.session_state['current_chat_history'][:-1]):
        st.markdown(f'<p><span style="color:#ADD8E6;">Query:</span> {chat["query"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p><span style="color:#ADD8E6;">Response:</span> {chat["response"]}</p>', unsafe_allow_html=True)
        st.write("---")
    

def main():
    st.header("Chat with Multiple PDF using Gemini💁")

    # Initialize the database
    init_db()

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Get Answer"):
        if user_question:
            user_input(user_question)
            show_current_chat_history()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

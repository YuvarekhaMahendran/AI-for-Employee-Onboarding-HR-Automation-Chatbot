import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def extract_text_from_pdfs(pdf_files):
    """Extracts text from uploaded PDF files."""
    return "".join(page.extract_text() for pdf in pdf_files for page in PdfReader(pdf).pages)


def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits extracted text into manageable chunks for processing."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap, length_function=len)
    return splitter.split_text(text)


def create_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def initialize_conversation_chain(vector_store):
    """Initializes a conversational retrieval chain using an LLM and vector store."""
    llm = ChatOpenAI(openai_api_key="#open ai key is used here")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)


def process_user_input(user_query):
    """Handles user input and updates chat history."""
    response = st.session_state.conversation({'question': user_query})
    st.session_state.chat_history = response['chat_history']

    for idx, message in enumerate(st.session_state.chat_history):
        template = user_template if idx % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="HR Onboarding And Automation", page_icon="📝")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("HR Onboarding And Automation 📝")
    user_query = st.text_input("Ask a question:")
    if user_query:
        process_user_input(user_query)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_pdfs = st.file_uploader("Upload your PDFs and click 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                extracted_text = extract_text_from_pdfs(uploaded_pdfs)
                text_chunks = split_text_into_chunks(extracted_text)
                vector_store = create_vector_store(text_chunks)
                st.session_state.conversation = initialize_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
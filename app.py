import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.text_splitter import SemanticChunker
from htmlTemplates import css

@st.cache_resource(ttl=300)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

@st.cache_resource(ttl=300)
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversattion_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversattion_chain

def handle_user_input(user_question):
    if "conversation" not in st.session_state:
        st.write("Please process the documents first.")
        return

    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                 st.markdown(message.content)

        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot with multiple PDFs",
                       page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChatBot with multiple PDFs")
    user_question = st.chat_input("Ask a question about your documents")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        # st.sidebar.text_input('OpenAI API Key', type='password')
        st.header("Add your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # create a vector store
                    vector_store = get_vector_store(text_chunks)
                    # create a conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                else:
                    st.warning("Add at least 1 PDF is required!!")

if __name__ == '__main__':
    main()

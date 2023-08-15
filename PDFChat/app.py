import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from  langchain.callbacks import get_openai_callback
import pickle
# sidebar contents
from dotenv import load_dotenv
import os

load_dotenv()

with st.sidebar:
    st.title('LLM chat app')
    st.markdown('''
                ## About
                This app is an LLM-powered chatbot built using:
                - [Streamlit](https://streamlit.io/)
                - [Langchain](http://pytho.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM Model
                ''')
    add_vertical_space(5)
    st.write('Made by Nilesh Sawant')


def main():
    st.header('Chat with PDF')

    # upload a PDF file 
    pdf = st.file_uploader('Upload your PDF file',type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)


        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings were loaded from disk')
        else:
                #enbedding
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(VectorStore, f)
                st.write('Embeddings computed')

        
    query = st.text_input("Please enter your query")
    st.write(query)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(chain_type="stuff",llm=llm)

        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            st.write(response)
            st.write(cb)

if __name__ == '__main__':
    main()
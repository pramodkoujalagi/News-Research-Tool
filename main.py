import os
import streamlit as st
import dill
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from requests.exceptions import RequestException
from embeddings import TfidfEmbeddings

load_dotenv()

st.title("CoolBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_tfidf.pkl"

main_placeholder = st.empty()
llm = ChatGroq(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        # load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        # create embeddings and save it to FAISS index
        embeddings = TfidfEmbeddings()
        texts = [doc.page_content for doc in docs if isinstance(doc.page_content, str)]
        
        if not texts:
            raise ValueError("No valid text content found in the documents.")
        
        embeddings.fit(texts)
        
        # Create FAISS index
        vectorstore_tfidf = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index using dill
        with open(file_path, "wb") as f:
            dill.dump(vectorstore_tfidf, f)
        
        main_placeholder.success("Processing completed successfully!")
    except RequestException as e:
        main_placeholder.error(f"Error loading URLs: {str(e)}")
        st.stop()
    except Exception as e:
        main_placeholder.error(f"An error occurred: {str(e)}")
        st.exception(e)
        st.stop()

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = dill.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain.invoke({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
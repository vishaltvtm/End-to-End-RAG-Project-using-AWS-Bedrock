import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following pieces of context to answer the question at the end.\n\n{context}\n\nQuestion: {question}\nAnswer:",
)
# Initialize the Bedrock client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",  # Change to your desired region
)

# Initialize the embeddings and LLM
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1",  # Change to your desired model
)

def get_document_embeddings():
    loader = PyPDFDirectoryLoader("data/pdfs")
    documents=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store():
    docs = get_document_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("vector_store/faiss_index")
    return vector_store

def get_retrieval_qa():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Bedrock(model_id="amazon.titan-text-instruct-v1", client=bedrock_client)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

Qa = get_retrieval_qa()
answer = Qa({ "query": "What is the capital of France?" })
import json
st.write(json.dumps(answer['result'], indent=2)) 

def main():
    st.title("LangChain with Amazon Bedrock")
    st.write("This app demonstrates how to use LangChain with Amazon Bedrock for document retrieval and question answering.")

    query = st.text_input("Enter your question:")
    if query:
        answer = Qa({"query": query})
        st.write("Answer:", answer['result'])
        if 'source_documents' in answer:
            st.write("Source Documents:")
            for doc in answer['source_documents']:
                st.write(doc.metadata.get('source', 'Unknown Source'))

if __name__ == "__main__":
    main()

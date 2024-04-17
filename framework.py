import dotenv
import os
dotenv.load_dotenv()
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    return documents

def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_data = text_splitter.split_documents(documents)
    return chunked_data

def store_data(directory_path,db_path):
    documents = load_documents(directory_path)
    chunked_data = split_document(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db_chroma = Chroma.from_documents(chunked_data, embeddings, persist_directory=db_path)
    return db_chroma

db = store_data('C:\Prithvi\RAG POC\Trignometry docs', 'C:\Prithvi\RAG POC\db_chroma')

template = """Use the following  context to answer the question at the end. 
    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer.
    {context}

    Question: {question}
    Answer: 
    """

def prompt_template(template):
    prompt = PromptTemplate.from_template(template = template)
    return prompt
 
def qa_chain(question):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature = 0.2)
    qa = RetrievalQA.from_llm(llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
                            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                            return_source_documents=False)
    response = qa({"query":question})
    return response['result']

def get_response(question):
    prompt = prompt_template(template)
    response = qa_chain(question)
    return response
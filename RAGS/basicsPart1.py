#define the directory containing text file and persisting directory
# check if the chroma vector store already exists. -> split the document in chunks ->  creating embeddings
# ->create vector store and persist it automation


import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#define the directory containing text file and persisting directory

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "tbateVOL8.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# check if the chroma vector store already exists.

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    #ensure if text file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path"
        )
    
    #Read the content form the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document in chunks
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #Display chunks information
    print("\n-- Document chunks information ---")
    print(f"Number of document chunks {len(docs)}")
    print(f"Simple chunk:/n{docs[0].page_content}")

    #creating embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished embedding")

    #create vector store and persist it automation
    print("\n--- creating vector store")
    db = Chroma.from_documents(
        docs,embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store")

else:
    print("Vector store already exist. No need to initialize")


    

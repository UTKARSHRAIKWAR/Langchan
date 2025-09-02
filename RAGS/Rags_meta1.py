#define the directory containing text file and persisting directory
# check if the chroma vector store already exists. -> split the document in chunks ->  creating embeddings
# ->create vector store and persist it automation


import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#define the directory containing text file and persisting directory

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_metadata")

print("\n--- Books directory {books_dir}")
print("\n--- Persistent directory {persistent_directory}")

# check if the chroma vector store already exists.

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    #ensure if text file exists.
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check the path"
        )
    
    #list all the text file in directory
    books_file = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    #Read the content form each file and store it with meta data
    documents = []
    for book_file in books_file:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        books_doc = loader.load()
        for doc in books_doc:
            #add meta data for each document indicating its source
            doc.metadata= {"source":book_file}
            documents.append(doc)

    # Split the document in chunks
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #Display chunks information
    print("\n-- Document chunks information ---")
    print(f"Number of document chunks {len(docs)}")
    print(f"Simple chunk:/n{docs[0].page_content}")

    #creating embeddings
    print("\n--- Creating embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("\n--- Finished embedding")

    #create vector store and persist it automation
    print("\n--- creating vector store")
    db = Chroma.from_documents(
        docs,embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store")

else:
    print("Vector store already exist. No need to initialize")


    

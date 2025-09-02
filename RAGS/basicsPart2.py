import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()



#define persistence directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db" , "chroma_db")


# define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# load the vector store with the embedding
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


#define the user question
query = "Arthur leywin ?"


#retrieve the relevant doc based on the query
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold":0.5}
)
relevant_docs = retriever.invoke(query)

#display the relevant results with metadata
print("\n ---Relevant documents")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n {doc.page_content}\n")
    if doc.metadata:
        print(f"source: {doc.metadata.get('source','Unknown')}\n")
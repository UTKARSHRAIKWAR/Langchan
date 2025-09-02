import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

#define persistence directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir , "chroma_db_metadata")


# define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# load the vector store with the embedding
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


#define the user question
query = "who is Klein Moretti ?"


#retrieve the relevant doc based on the query
retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)
relevant_docs = retriever.invoke(query)

# Rag one off question
#combined the query and relevant doc content
combined_query = (
    "Here are some documents that might help answer the question: " +
    query +
    "\n\n Relevant documents:\n" +
    "\n\n".join([doc.page_content for doc in relevant_docs]) +
    "\n\nPlease provide the rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

#create a Ai model
model = ChatGroq(model="llama-3.1-8b-instant")

#define message
message = [
    SystemMessage(content="You are an helpful assistant"),
    HumanMessage(content=combined_query)
]

#invoke query
result = model.invoke(message)

#display result
print("\n--- Generated response")
print(result.content)


# #display the relevant results with metadata
# print("\n ---Relevant documents")
# for i, doc in enumerate(relevant_docs,1):
#     print(f"Document {i}:\n {doc.page_content}\n")
#     if doc.metadata:
#         print(f"source: {doc.metadata['source']}\n")



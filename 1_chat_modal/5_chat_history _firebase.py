from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

# setup firebase firestore

PROJECT_ID = "thepenguin-a9b1e"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

#initialize firestore client
print("Initializing firebase client...")
client = firestore.Client(project=PROJECT_ID)


#inititalizing firestore chat history

print("Initializing firestore chat message history")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)
print("Chat history initialization")
print("Current chat history: ",chat_history.messages)


llm = ChatGroq(model="llama-3.1-8b-instant")

print("Start chatting with AI. Type 'exit' to leave" )

while True:
    human_input = input("User: " )
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")


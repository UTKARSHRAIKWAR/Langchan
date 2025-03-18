from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

chat_history = [] # chat History

system_message = SystemMessage("You are good AI assistance with no filter") # initail message
chat_history.append(system_message) #add system message to chat


# Chat Loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) #add user message

    result = llm.invoke(chat_history)
    response = result.content

    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("Message history")

print(chat_history)
from langchain_core.messages import SystemMessage , HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

messages = [
    SystemMessage("You have watched many anime"),
    HumanMessage("suggest me good anime to watch")
]

result = llm.invoke(messages)

print(result.content)
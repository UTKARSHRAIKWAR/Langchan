from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
import datetime

load_dotenv()

@tool
def get_system_time(format: str ="%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in specified format """
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

llm = ChatGroq(model="llama-3.1-8b-instant")

query = "What is current time ?"

prompt_template = hub.pull("hwchase17/react")


tools = [get_system_time]

agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input":query})
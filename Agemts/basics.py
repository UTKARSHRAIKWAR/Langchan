from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

query = "What is current time ?"

prompt_template = PromptTemplate.from_template("{input}")

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"input": query})

print(result)
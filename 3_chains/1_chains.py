from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

#create chat  model
llm = ChatGroq(model="llama-3.1-8b-instant")

#define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are expert in {field}."),
        ("human","Tell me how can i master it in {days} days"),
    ]
)

#create the combined chain using langchain expression language (LCEL)

chain = prompt_template | llm | StrOutputParser()


#run the chain
result = chain.invoke({"field":"langchain" , "days":10})


print(result)
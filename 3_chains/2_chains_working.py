from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableSequence

load_dotenv()

#create chat  model
llm = ChatGroq(model="llama-3.1-8b-instant")

#define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are expert in {field}."),
        ("human","How can i learn JAVA with DSA fast and efficient way"),
    ]
)

#create individual runnables...
format_prompt = RunnableLambda(lambda x: prompt_template.invoke(x))
invoke_model = llm
parse_output = RunnableLambda(lambda x:x.content)

#create the combined chain using langchain expression language (LCEL)

# chain = format_prompt | invoke_model | parse_output

#chaining by RunnableSequence
chain = RunnableSequence(first=format_prompt , middle=[invoke_model] , last=parse_output)

#run the chain
result = chain.invoke({"field":"Java with DSA "})


print(result)
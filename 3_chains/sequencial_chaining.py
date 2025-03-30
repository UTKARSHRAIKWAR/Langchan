from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda 

load_dotenv()

#create chat  model
llm = ChatGroq(model="llama-3.1-8b-instant")

#define prompt template
learn_dsa_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are expert in {field}."),
        ("human","How can i learn {language} with DSA fast and efficient way"),
    ]
)

translate_content_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a translator and convert the provided text into {language}"),
        ("human","Translate the following content to {language}:{text}")
    ]
)

count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")
prepareForTranslation = RunnableLambda(lambda output: {"text":output, "language":"highish"})


chain = learn_dsa_template | llm | StrOutputParser() | prepareForTranslation | translate_content_template | llm | StrOutputParser()

result = chain.invoke({"field":"DSA" , "language":"JAVA"})

print(result)
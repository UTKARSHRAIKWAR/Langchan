from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableParallel , RunnableBranch

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

#define prompt template for different feedback

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant"),
        ("human","Generate a thankyou note for this positive feedback:{feedback}"),
    ]
)
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant"),
        ("human","Generate a response addressing this negative feedback:{feedback}"),
    ]
)
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant"),
        ("human","Generate a request for asking more details for this neutral feedback:{feedback}"),
    ]
)
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistant"),
        ("human","Generate a message to escalate this feedback to human agent :{feedback}"),
    ]
)

#define a feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an helpful assistance"),
        ("human","Classify the sentiment of this feedback as Positive, negative, neutral, escalate: {feedback}")
    ]
)

#define runnable branches
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | llm | StrOutputParser()
    ),
    escalate_feedback_template | llm | StrOutputParser()
)

classification_chain = classification_template | llm | StrOutputParser()

chain = classification_chain | branches

review = "This is terrible product"

result = chain.invoke({"feedback":review})

print(result)
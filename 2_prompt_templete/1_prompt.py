from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

template = "Write a {tone} email to {company} for internship opportunity , mentioning {skill} as a key strength. And make it short and good. "

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone":"energetic",
    "company":"TCS",
    "skill":"AI engineer and full stack development"
})

result = llm.invoke(prompt)

print(prompt)

print(result.content)
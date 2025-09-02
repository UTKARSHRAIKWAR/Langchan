from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableParallel

load_dotenv()

#create chat  model
llm = ChatGroq(model="llama-3.1-8b-instant")

#define prompt template
summary_template = ChatPromptTemplate.from_messages(
    [
       ("system","You are a movie/critic critic"),
       ("human","Provide a brief summaryof the movie {movie_name}")
    ]
)

#define plot analyse step
def analysePlot(plot) :
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie/critic critic"),
            ("human","Analyse the plot:{plot}. What are its weakness and strengths?")
        ]
    )
    return plot_template.format_prompt(plot=plot)

#define character analyse step
def analyseCharacter(character):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie/series critic"),
            ("human","Analyse the characters: {character}. What are their strength and weakness?")
        ]
    )
    return character_template.format_prompt(character=character)

#combine analysis into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot analysis:\n{plot_analysis}\n\nCharacter analysis:\n{character_analysis}"

#simplify branches with LCEL
plot_chain_branch = (
    RunnableLambda(lambda x: analysePlot(x)) | llm | StrOutputParser()
)

character_chain_branch = (
    RunnableLambda(lambda x: analyseCharacter(x)) | llm | StrOutputParser()
)

chain = (
    summary_template
    | llm 
    | StrOutputParser()
    | RunnableParallel(branches={"plot":plot_chain_branch , "character":character_chain_branch})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"],x["branches"]["character"]))
)

result = chain.invoke({"movie_name":"inception"})

print(result)
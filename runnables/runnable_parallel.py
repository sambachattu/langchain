from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

promt1 = PromptTemplate(
    template="generate a tweet about {topic}",
    input_variables=['topic']
)

promt2 = PromptTemplate(
    template='Generate a linkedIn post about {topic}',
    input_variables=['topic']
)

model  = ChatOpenAI()

parser  = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(promt1,model,parser),
    'linkedin':RunnableSequence(promt2,model,parser)
})

result = parallel_chain.invoke({'topic':'AI'})
print(result)
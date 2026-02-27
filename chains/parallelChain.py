from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model1 = ChatOpenAI()

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    input_types='Generate short note from the following text\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='generate 5 question answer from the following text \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='merge the provided note and quiz into a single document \n note -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

prallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = prallel_chain | merge_chain

text = """ 
Embedding models transform raw text—such as a sentence, paragraph, or tweet—into a fixed-length vector of numbers that captures its semantic meaning. These vectors allow machines to compare and search text based on meaning rather than exact words.
In practice, this means that texts with similar ideas are placed close together in the vector space. For example, instead of matching only the phrase “machine learning”, embeddings can surface documents that discuss related concepts even when different wording is used.
​
How it works
Vectorization — The model encodes each input string as a high-dimensional vector.
Similarity scoring — Vectors are compared using mathematical metrics to measure how closely related the underlying texts are.
​
Similarity metrics
Several metrics are commonly used to compare embeddings:
Cosine similarity — measures the angle between two vectors.
Euclidean distance — measures the straight-line distance between points.
Dot product — measures how much one vector projects onto another.
Here’s an example of computing cosine similarity between two vectors:
import numpy as np

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(query_embedding, document_embedding)
print("Cosine Similarity:", similarity)
"""

result = chain.invoke({'text':text})

print(result)
# MMR picks results that are not only relavent but also different from eachother
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs={"k":3,"lambda_mult":1} # k = top results, lambda_mult = relevance-diversity balance
)

query = "What is langchain"

result = retriever.invoke(query)

for i,doc in enumerate(result):
    print(f"\nResult {i+1}")
    print(doc.page_content)

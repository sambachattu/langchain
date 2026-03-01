from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = OpenAIEmbeddings()

vectore_store = Chroma.from_documents(
    embedding = embedding_model,
    collection_name='my_collection',
    documents = documents
)

retriever = vectore_store.as_retriever(search_kwargs={'k':2})

query = "What is Chroma used for"

result = retriever.invoke(query)

for i,doc in enumerate(result):
    print(f"\n---Result {i+1}")
    print(doc.page_content)


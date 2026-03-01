from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2,lang='en')

query = "About ramanujan"

docs = retriever.invoke(query)

for i,doc in enumerate(docs):
    print(f"Result{i+1}")
    print(f"Content:\n{doc.page_content}")
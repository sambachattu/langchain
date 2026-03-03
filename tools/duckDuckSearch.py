from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
load_dotenv()
search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke('Top news in india')

print(result)
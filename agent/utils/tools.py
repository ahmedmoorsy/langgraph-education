from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages

# Initialize the Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o")

# Initialize the message trimmer
trimmer = trim_messages(
    max_tokens=100000,
    strategy="last",
    token_counter=llm,
    include_system=True,
)
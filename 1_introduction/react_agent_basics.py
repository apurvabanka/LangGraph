from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

search_tools = TavilySearchResults(search_depth='basic')

@tool
def get_system_time(input: str) -> str:
    """Returns the current system time in a human-readable format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_tools, get_system_time]

agent = initialize_agent(tools=tools,llm=llm, agent='zero-shot-react-description', verbose=True)

agent.invoke("when was the SpaceX's last launch and how long ago was it?")





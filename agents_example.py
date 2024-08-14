from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain.chains import create_retrieval_chain
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. "),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Given is a quest name {quest}. "
              "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} and find the previous quests. "
              "Let's say the previous quest is 'some_quest'. Then, find the previous quest"
              "of 'some_quest', and continue this workflow, until 10 recursive previous quests are found (if exists). "
              "Give me then previous quests which is found."),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)
chain = create_retrieval_chain()

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#command = {
#    "input": "Go to https://python.langchain.com/v0.2/docs/integrations/toolkits/playwright/ "
#             "and give me summary of all tools mentioned on the page you get. Print out url at each step."
#}
command = {
    "quest": "On Rough Seas"
}
agent_executor.invoke(command)

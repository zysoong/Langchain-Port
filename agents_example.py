from typing import Sequence

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI


def create_openai_rag_tools_agent(llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate) \
        -> Runnable:
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    def create_user_message_chain(x):
        return format_to_openai_tool_messages(x["intermediate_steps"])

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=create_user_message_chain
            )
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
    )
    return agent


if __name__ == '__main__':
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. "),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "Given is a quest name {quest}. "
                  "Go to https://ffxiv.consolegameswiki.com/wiki/{quest} and find the previous quests. "
                  "Let's say the previous quest is 'some_quest'. Then, find the previous quest"
                  "of 'some_quest', and continue this workflow, until 10 recursive previous quests are found (if "
                  "exists)."
                  "Give me all previous quests which is found."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    sync_browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_openai_rag_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)

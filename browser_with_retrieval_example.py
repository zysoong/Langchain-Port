from typing import Sequence

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
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

from tool.injector import PromptInjector
from tool.toolkit import RetrievalPlayWrightBrowserToolkit


def create_openai_tools_agent_and_inject_prompts(
        llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses OpenAI tools.
    The prompts will be injected to the tools
    automatically.
    Check the documentation of 'create_openai_tools_agent'
    for detailed instructions.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to. Prompts will be injected as 'prompt'
            attribute automatically.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | prompt
            | PromptInjector(inject_objects=tools, pass_on_injection_fail=True)
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
    toolkit = RetrievalPlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = create_openai_tools_agent_and_inject_prompts(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    command = {
        "quest": "On_Rough_Seas"
    }
    agent_executor.invoke(command)

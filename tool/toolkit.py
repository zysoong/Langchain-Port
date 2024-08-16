from typing import List, Type, cast

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools import ClickTool, NavigateTool, NavigateBackTool, ExtractTextTool, ExtractHyperlinksTool, \
    GetElementsTool, CurrentWebPageTool
from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_core.tools import BaseTool
from tool.extract_text import RetrievalExtractTextTool


class RetrievalPlayWrightBrowserToolkit(PlayWrightBrowserToolkit):
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tool_classes: List[Type[BaseBrowserTool]] = [
            ClickTool,
            NavigateTool,
            NavigateBackTool,
            RetrievalExtractTextTool,
            ExtractHyperlinksTool,
            GetElementsTool,
            CurrentWebPageTool,
        ]

        tools = [
            tool_cls.from_browser(
                sync_browser=self.sync_browser, async_browser=self.async_browser
            )
            for tool_cls in tool_classes
        ]
        return cast(List[BaseTool], tools)

from __future__ import annotations

from typing import Optional, Type

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain_community.tools.playwright.utils import (
    aget_current_page, get_current_page,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


class RetrievalExtractTextTool(BaseBrowserTool):
    """Tool for extracting and retrieve relevant text on the current webpage."""

    name: str = "retrieval_extract_text"
    description: str = "Extract and retrieve relevant text on the current webpage"
    args_schema: Type[BaseModel] = BaseModel

    @root_validator(pre=True)
    def check_acheck_bs_importrgs_for_ragextract(cls, values: dict) -> dict:
        """Check that the arguments are valid."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required to use this tool."
                " Please install it with 'pip install beautifulsoup4'."
            )
        return values

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        # Use Beautiful Soup since it's faster than looping through the elements
        from bs4 import BeautifulSoup

        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")

        page = get_current_page(self.sync_browser)
        html_content = page.content()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        # Retrieve relevant text
        full_text = "".join(text for text in soup.stripped_strings)
        document = Document(page_content=full_text, metadata={"source": "local"})
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents([document])
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents("Get relevant text of user request")
        return " ".join(doc.page_content for doc in docs)

    async def _arun(
            self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        # Use Beautiful Soup since it's faster than looping through the elements
        from bs4 import BeautifulSoup

        page = await aget_current_page(self.async_browser)
        html_content = await page.content()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        # Retrieve relevant text
        full_text = "".join(text for text in soup.stripped_strings)
        document = Document(page_content=full_text, metadata={"source": "local"})
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents([document])
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents("Get relevant text of user request")
        return " ".join(doc.page_content for doc in docs)

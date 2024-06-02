from typing import List, Optional, Any, Dict

import logging
from agents.specs import ChatCompletion
from agents.tool_executor import ToolRegistry
from langchain_core.tools import StructuredTool
from llama_cpp import ChatCompletionRequestMessage
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIChatCompletion:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()
        self.tool_registry = ToolRegistry()

    def bind_tools(self, tools: Optional[List[StructuredTool]] = None):
        for tool in tools:
            self.tool_registry.register_tool(tool)

    def chat_completion(
        self, messages: List[ChatCompletionRequestMessage], **kwargs
    ) -> ChatCompletion:
        tools = self.tool_registry.openai_tools
        output = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools
        )
        logger.debug(output)
        return output

    def run_tools(self, chat_completion: ChatCompletion) -> List[Dict[str, Any]]:
        return self.tool_registry.call_tools(chat_completion)

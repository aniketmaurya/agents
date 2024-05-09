import time

from langchain_cohere import ChatCohere

from typing import List, Optional, Any, Dict

from langchain_core.messages import AIMessage
import logging
from agents.specs import ChatCompletion, Choice, Usage, Message, ToolCall
from agents.tool_executor import ToolRegistry
from langchain_core.tools import StructuredTool
from llama_cpp import ChatCompletionRequestMessage

logger = logging.getLogger(__name__)


class CohereChatCompletion:
    def __init__(self, **kwargs):
        self.llm = ChatCohere()
        self.tool_registry = ToolRegistry()

    def bind_tools(self, tools: Optional[List[StructuredTool]] = None):
        self.llm = self.llm.bind_tools(tools)
        for tool in tools:
            self.tool_registry.register_tool(tool)

    def _format_cohere_to_openai(self, output: AIMessage):
        response_metadata = output.response_metadata
        _tool_calls = response_metadata.get("tool_calls", [])
        tool_calls = []
        for tool in _tool_calls:
            tool = ToolCall(id=tool["id"], type=tool["type"], function=tool["function"])
            tool_calls.append(tool)

        message = Message(
            role="assistant", content=output.content, tool_calls=tool_calls
        )
        choices = Choice(index=0, logprobs=None, message=message, finish_reason="stop")
        usage = Usage(
            prompt_tokens=response_metadata["token_count"].get("input_tokens", -1),
            completion_tokens=response_metadata["token_count"]["output_tokens"],
            total_tokens=-1,
        )
        response = {
            "id": output.id,
            "object": "",
            "created": int(time.time()),
            "model": "Cohere",
            "choices": [choices],
            "usage": usage,
        }
        return ChatCompletion(**response)

    # def _format_cohere_to_openai(self, output: NonStreamedChatResponse):
    #     _tool_calls = output.tool_calls
    #     tool_calls = []
    #     for tool in _tool_calls:
    #         tool = ToolCall(id=tool["id"], type=tool["type"], function=tool["function"])
    #         tool_calls.append(tool)
    #
    #     message = Message(
    #         role="assistant", content=output.text, tool_calls=tool_calls
    #     )
    #     choices = Choice(index=0, logprobs=None, message=message, finish_reason=output.finish_reason)
    #     usage = Usage(
    #         prompt_tokens=output.meta.tokens.input_tokens,
    #         completion_tokens=output.meta.tokens.output_tokens,
    #         total_tokens=(output.meta.tokens.input_tokens + output.meta.tokens.output_tokens),
    #     )
    #     response = {
    #         "id": output.generation_id,
    #         "object": "",
    #         "created": int(time.time()),
    #         "model": "Cohere",
    #         "choices": [choices],
    #         "usage": usage,
    #     }
    #     return ChatCompletion(**response)

    def chat_completion(
        self, messages: List[ChatCompletionRequestMessage], **kwargs
    ) -> ChatCompletion:
        output = self.llm.invoke(messages, **kwargs)
        logger.debug(output)
        return self._format_cohere_to_openai(output)

    def run_tools(self, chat_completion: ChatCompletion) -> List[Dict[str, Any]]:
        return self.tool_registry.call_tools(chat_completion)

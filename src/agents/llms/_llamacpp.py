from typing import List, Optional, Any, Dict
from loguru import logger

from agents.specs import ChatCompletion
from agents.tool_executor import ToolRegistry
from langchain_core.tools import StructuredTool
from llama_cpp import ChatCompletionRequestMessage
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from llama_cpp import Llama


def create_tool_use_llm(**kwargs):
    return Llama.from_pretrained(
        repo_id="meetkai/functionary-small-v2.2-GGUF",
        filename="functionary-small-v2.2.q4_0.gguf",
        chat_format="functionary-v2",
        tokenizer=LlamaHFTokenizer.from_pretrained(
            "meetkai/functionary-small-v2.2-GGUF"
        ),
        **kwargs,
    )


class LlamaCppChatCompletion:
    model: Llama

    def __init__(self, model_or_path, **kwargs):
        if isinstance(model_or_path, str):
            self.model = Llama.from_pretrained(model_or_path, **kwargs)
        elif isinstance(model_or_path, Llama):
            self.model = model_or_path

        else:
            raise ValueError(f"Unsupported model type: {type(model_or_path)}")

        self.tool_registry = ToolRegistry()

    @staticmethod
    def from_default_llm(verbose=False, n_gpu_layers=-1, **kwargs):
        """Kwargs for `LlamaCppChatCompletion.from_pretrained`"""
        return LlamaCppChatCompletion(
            create_tool_use_llm(n_gpu_layers=n_gpu_layers, verbose=verbose, **kwargs)
        )

    def bind_tools(self, tools: Optional[List[StructuredTool]] = None):
        if tools:
            for tool in tools:
                self.tool_registry.register_tool(tool)

    @property
    def tools(self):
        return self.tool_registry.openai_tools

    def chat_completion(
        self, messages: List[ChatCompletionRequestMessage], **kwargs
    ) -> ChatCompletion:
        output = self.model.create_chat_completion(messages, tools=self.tools, **kwargs)
        logger.debug(output)
        return ChatCompletion(**output)

    def run_tools(self, chat_completion: ChatCompletion) -> List[Dict[str, Any]]:
        return self.tool_registry.call_tools(chat_completion)

"""Given the following response from LLM, call the tool.

```
{'id': 'chatcmpl-339bfd4b-f22c-45a2-9265-40906eadafa9',
 'object': 'chat.completion',
 'created': 1715032029,
 'model': 'Llama',
 'choices': [{'index': 0,
   'logprobs': None,
   'message': {'role': 'assistant',
    'content': None,
    'tool_calls': [{'id': 'call_YKQmVTO2Alwas834Yzf1UpJp',
      'type': 'function',
      'function': {'name': 'get_current_weather',
       'arguments': '{"city": "London"}'}}]},
   'finish_reason': 'tool_calls'}],
 'usage': {'prompt_tokens': 200, 'completion_tokens': 16, 'total_tokens': 201}}
```

######### Steps ##########
1. Check if LLM suggested tool use.
2. Check if the function signatures are valid
3. Check if function is available
4. Check if function can be called with the provided arguments
5. Execute the function with given arguments
6. Return the result of the function or raise the proper exception


########## Notes ########
Step 2 is auto checked with Pydantic
"""

import json
from typing import Any, List

from langchain_community.tools import StructuredTool
import logging

from langchain_core.utils.function_calling import convert_to_openai_function

from agents.specs import ChatCompletion, ToolCall


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, StructuredTool] = {}
        self._openai_tools: dict[str, Any] = {}

    def register_tool(self, tool: StructuredTool):
        self._tools[tool.name] = tool
        self._openai_tools[tool.name] = convert_to_openai_function(tool)

    def get(self, name: str) -> StructuredTool:
        return self._tools.get(name)

    def __getitem__(self, name: str):
        return self._tools[name]

    def pop(self, name: str) -> StructuredTool:
        return self._tools.pop(name)

    @property
    def openai_tools(self) -> List[dict[str, Any]]:
        # [{"type": "function", "function": registry.openai_tools[0]}],
        result = []
        for oai_tool in self._openai_tools.values():
            result.append({"type": "function", "function": oai_tool})

        return result

    def call_tool(self, output: ChatCompletion | dict) -> list[dict[str, str]]:
        if isinstance(output, dict):
            output = ChatCompletion(**output)

        if not has_tool_use(output):
            raise ValueError(f"No tool call was found in ChatCompletion\n{output}")

        messages = []
        # https://platform.openai.com/docs/guides/function-calling
        tool_calls = output.choices[0].message.tool_calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.get(function_name)

            if not function_to_call:
                raise ValueError(f"No function was found for {function_name}")

            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call.invoke(function_args)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        return messages


def has_tool_use(output: ChatCompletion):
    tool_calls = output.choices[0].message.tool_calls
    if tool_calls:
        return True
    return False


def check_function_signature(
    output: ChatCompletion, tool_registry: ToolRegistry = None
):
    tools = output.choices[0].message.tool_calls
    invalid = False
    for tool in tools:
        tool: ToolCall
        if tool.type == "function":
            function_info = tool.function
            if tool_registry:
                if tool_registry.get(function_info.name) is None:
                    logging.error(f"Function {function_info.name} is not available")
                    invalid = True

            arguments = function_info.arguments
            try:
                json.loads(arguments)
            except json.JSONDecodeError as e:
                logging.exception(e)
                invalid = True
        if invalid:
            return False

    return True

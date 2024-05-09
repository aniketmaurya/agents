from agents.specs import ChatCompletion
from pydantic import BaseModel
from agents.tools import get_current_weather
from agents.tool_executor import ToolRegistry
import json

sample = '{"id":"chatcmpl-4483920d-3cb8-46c1-9c3a-d2eb0f320e59","object":"chat.completion","created":1715036883,"model":"/Users/aniket/.cache/huggingface/hub/models--meetkai--functionary-small-v2.2-GGUF/snapshots/9971ff5d1eacbfac81fe6939bccd59cf7d668078/./functionary-small-v2.2.q4_0.gguf","choices":[{"index":0,"logprobs":null,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_DnmopdelmY8Dl1NRXXx2gMDy","type":"function","function":{"name":"get_current_weather","arguments":"{\\"city\\": \\"San Francisco\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":201,"completion_tokens":17,"total_tokens":202}}'

completion_data = json.loads(sample)


def test_registry():
    tool_registry = ToolRegistry()
    tool_registry.register_tool(get_current_weather)
    assert tool_registry.get("get_current_weather")

    messages = tool_registry.call_tools(completion_data)
    assert "FeelsLikeC" in messages[0]["content"]


def test_completion():
    completions = ChatCompletion(**completion_data)
    assert isinstance(completions, BaseModel)

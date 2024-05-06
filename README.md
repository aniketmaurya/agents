# Agents 🤖

<!-- A brief description of what this project does and who it's for -->
A fun project to build Agentic workflows with function calling, powered by LangChain.


## Installation
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/aniketmaurya/python-project-template?template=false)

Install Agents with pip

```bash
  git clone https://github.com/aniketmaurya/agents.git
  cd agents
  pip install .
```

## Usage/Examples

```python
from langchain_core.utils.function_calling import convert_to_openai_function
from rich import print

from agents.specs import ChatCompletion
from agents.tool_executor import ToolRegistry
from agents.llm import create_tool_use_llm
from agents.tools import get_current_weather

llm = create_tool_use_llm(verbose=False, n_gpu_layers=-1)

registry = ToolRegistry()
registry.register_tool(get_current_weather)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "How is the weather in SF today?"}],
    tools=[{"type": "function", "function": registry.openai_tools[0]}],
)

output = ChatCompletion(**output)
tool_response = registry.call_tool(output)
print(tool_response)
```


<!-- ## Demo

Insert gif or link to demo -->


<!-- ## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2 -->

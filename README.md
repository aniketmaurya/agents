# Agents 🤖

Build Agentic workflows with function calling, powered by LangChain.

## Installation

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/aniketmaurya/python-project-template?template=false)

Install Agents with pip

```bash
  git clone https://github.com/aniketmaurya/agents.git
  cd agents
  pip install .
```

## Usage/Examples

LLM with access to weather API:

```python
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

<details>
    <summary>See output:</summary>

```shell
[{'tool_call_id': 'call_DnmopdelmY8Dl1NRXXx2gMDy',
  'role': 'tool',
  'name': 'get_current_weather',
  'content': [{'FeelsLikeC': '17',
    'FeelsLikeF': '62',
    'cloudcover': '50',
    'humidity': '50',
    'localObsDateTime': '2024-05-06 03:35 PM',
    'observation_time': '10:35 PM',
    'precipInches': '0.0',
    'precipMM': '0.0',
    'pressure': '1021',
    'pressureInches': '30',
    'temp_C': '17',
    'temp_F': '62',
    'uvIndex': '4',
    'visibility': '16',
    'visibilityMiles': '9',
    'weatherCode': '116',
    'weatherDesc': [{'value': 'Partly cloudy'}],
    'weatherIconUrl': [{'value': ''}],
    'winddir16Point': 'W',
    'winddirDegree': '270',
    'windspeedKmph': '28',
    'windspeedMiles': '17'}]}]
```

</details>




<!-- ## Demo

Insert gif or link to demo -->


<!-- ## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2 -->

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
from agents.hermes.functioncall import ModelInference


# model_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
model_path = "/Users/aniket/weights/llama-cpp/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"

model_inference = ModelInference(
    model_path=model_path,
)
model_inference.generate_function_call(
    "When was Golden gate bridge painted last time?", None, 5
)
```


<!-- ## Demo

Insert gif or link to demo -->


<!-- ## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2 -->

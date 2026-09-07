<p align="center">
  <img width="250" alt="logo" src="https://ik.imagekit.io/gradsflow/logo/v2/gf-logo-gradsflow-orange_bv-f7gJu-up.svg"/>
  <br>
  <strong>AgentForce: A Production-Ready Framework for Building AI Agents</strong>
</p>

<p align="center">
  <a href="https://agents.gradsflow.com">Documentation</a> |
  <a href="https://github.com/gradsflow/agentforce/tree/main/examples">Examples</a> |
  <a href="#-quick-start">Quick Start</a>
</p>

---

AgentForce is a powerful, open-source framework designed for building production-ready AI agents. It simplifies the integration of various LLMs and tools, enabling you to create sophisticated AI applications with minimal setup.

## ✨ Key Features

- 🤖 **Multiple LLM Support**: OpenAI, Cohere Command (R/R+), and LlamaCPP integration
- 🛠️ **Tool Integration**: Seamlessly add capabilities like web search, weather data, and image analysis
- 👁️ **Multi-modal Support**: Process both text and images in your AI workflows
- 🚀 **Production Ready**: Built with scalability and reliability in mind
- 📦 **Easy to Extend**: Simple API for adding custom tools and LLM providers

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install agentforce

# Install latest from GitHub
pip install git+https://github.com/gradsflow/agentforce.git@main

# Development installation
git clone https://github.com/gradsflow/agentforce.git
cd agentforce
pip install -e .
```

### Basic Usage

Here's a simple example using AgentForce with weather data:

```python
from agentforce.llms import LlamaCppChatCompletion
from agentforce.tools import get_current_weather
from agentforce.tool_executor import need_tool_use

# Initialize LLM with weather tool
llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([get_current_weather])

# Create a simple query
messages = [{"role": "user", "content": "How is the weather in London today?"}]

# Get response and handle tool usage
output = llm.chat_completion(messages)

if need_tool_use(output):
    tool_results = llm.run_tools(output)
    updated_messages = messages + tool_results
    updated_messages.append({"role": "user", "content": "Summarize the weather information."})
    output = llm.chat_completion(updated_messages)

print(output.choices[0].message.content)
```

## 🎯 Advanced Features

### Multi-modal Agent Example

Create agents that can understand and process both text and images:

```python
from agentforce.llms import LlamaCppChatCompletion
from agentforce.tools import wikipedia_search, google_search, image_inspector

# Initialize LLM with multiple tools
llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([google_search, wikipedia_search, image_inspector])

# Process image and generate response
image_url = "https://example.com/image.jpg"
messages = [
    {"role": "system", "content": "You are a helpful assistant that can analyze images."},
    {"role": "user", "content": f"What can you tell me about this image? {image_url}"},
]

output = llm.chat_completion(messages)
tool_results = llm.run_tools(output)
final_output = llm.chat_completion(messages + tool_results)
```

## 📚 Documentation

For detailed documentation and advanced usage examples, visit our [Documentation](https://agents.gradsflow.com).

## 🤝 Contributing

We welcome contributions of all kinds! Whether it's:
- 📝 Improving documentation
- 🐛 Bug fixes
- ✨ New features
- 🔧 Tool integrations

Check out our [Contributing Guidelines](https://github.com/gradsflow/agentforce/blob/master/CONTRIBUTING.md) to get started.

## 📜 Code of Conduct

We are committed to fostering an open and welcoming environment. Please read our [Code of Conduct](https://github.com/gradsflow/agentforce/blob/master/CODE_OF_CONDUCT.md).

## 🙌 Acknowledgements

Built with ❤️ using **PyCharm**  
*Special thanks to JetBrains for their support!*

<div align="center">
  <img src="https://resources.jetbrains.com/storage/products/company/brand/logos/PyCharm_icon.svg" alt="PyCharm logo" width="100"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.svg" alt="JetBrains logo" width="100"/>
</div>

## 📄 License

This project is licensed under the [Apache License](LICENSE).

# Agents 🤖

Build Agentic workflows with function calling, powered by LangChain.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/aniketmaurya/python-project-template?template=false)

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/introduction-to-ai-agents">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>


## Installation

**Install latest branch:**

```shell
pip install git+https://github.com/aniketmaurya/agents.git@main
```

**or, for editable installation:**

```shell
  git clone https://github.com/aniketmaurya/agents.git
  cd agents
  pip install -e .
```

## Supported LLMs

✅ OpenAI

✅ Cohere Command R and Command R+

✅ LlamaCPP (open-source)

## Usage/Examples

### Simple tool use with a local or cloud LLM

LLM with access to weather API:

```python
from agents.llms import LlamaCppChatCompletion
from agents.tools import get_current_weather, wikipedia_search
from agents.tool_executor import need_tool_use

llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([get_current_weather, wikipedia_search])  # Add any tool from LangChain

messages = [
    {"role": "user", "content": "how is the weather in London today?"}
]

output = llm.chat_completion(messages)

if need_tool_use(output):
    print("Using weather tool")
    tool_results = llm.run_tools(output)
    tool_results[0]["role"] = "assistant"

    updated_messages = messages + tool_results
    updated_messages = updated_messages + [
        {"role": "user", "content": "Think step by step and answer my question based on the above context."}
    ]
    output = llm.chat_completion(updated_messages)

print(output.choices[0].message.content)
```

<details>
    <summary>Expand output:</summary>

```text
Certainly, let's break down the information provided in the weather data for London:

1. **Temperature**: It is currently 23°C (73°F) in London.
2. **Cloud Cover**: There are no clouds at the moment.
3. **Humidity**: The humidity level is 38%.
4. **Precipitation**: There has been no precipitation today, with 0 inches recorded.
5. **Pressure**: The atmospheric pressure is 1023 hPa (30 inches).
6. **Visibility**: The visibility is currently 10 km (6 miles).
7. **Weather Condition**: It's a sunny day in London.
8. **Wind**: The wind is blowing from the northwest at a speed of 9 km/h (6 mph).

Based on this information, it seems like today is a beautiful and sunny day in London. The temperature is quite
pleasant, and there's no precipitation to worry about. It's a great time to be outdoors!
```

</details>


`Agents` also support Cohere API for tool use/function calling. Check out the reproducible notebook [here](https://github.com/aniketmaurya/agents/blob/main/examples/cohere.ipynb).


<br>

### Multi-modal Agent 👁🤖️

You can build an AI Agent that can see the world with computer vision.

```python
from agents.llms import LlamaCppChatCompletion
from agents.tools import wikipedia_search, google_search, image_inspector

llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([google_search, wikipedia_search, image_inspector])

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
messages = [
    {"role": "system", "content": "You are a helpful assistant that has access to tools and use that to help humans."},
    {"role": "user",
     "content": f"Check this image {image_url} and suggest me a location where I can go in London which looks similar"}
]
output = llm.chat_completion(messages)
tool_results = llm.run_tools(output)

updated_messages = messages + tool_results
messages = updated_messages + [{"role": "user", "content": "please answer me, based on the tool results."}]
output = llm.chat_completion(messages)
print(output.choices[0].message.content)
```

<details>
    <summary>Expand output:</summary>

```text
Based on the image you provided, it appears to be a nature boardwalk surrounded by lush green grass and a peaceful
sky filled with clouds. The presence of people indicates that it could be a popular spot for outdoor activities or
leisurely walks. If you're looking for a similar location in London, you might consider visiting one of the city's
many parks or nature reserves. Here are a few suggestions:

1. **Richmond Park**: This is the largest royal park in London and offers a variety of landscapes, including open
grasslands, wooded areas, and lakes. It's a great place for walking, cycling, and enjoying the outdoors.

2. **Hampstead Heath**: Another large green space in London, Hampstead Heath is known for its ponds, meadows, and
woodlands. It's a popular spot for picnics, sunbathing, and hiking.

3. **Greenwich Park**: This park offers panoramic views of the city and is home to several historic buildings,
including the Royal Observatory. It's a great place for a leisurely walk or a picnic.

4. **Victoria Park**: A smaller but still beautiful park in East London, Victoria Park is known for its lakes,
gardens, and outdoor events.

5. **Hyde Park**: One of the most central parks in London, Hyde Park offers a variety of attractions, including the
Serpentine Lake, Speaker's Corner, and several monuments.

These locations all offer a peaceful and natural environment similar to the image you provided, making them
excellent choices for a day out in London.
```

</details>

## Acknowledgements

Built with PyCharm 🧡. Thanks to JetBrains for supporting this work by providing free credits.

<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/PyCharm_icon.svg" alt="PyCharm logo.">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.svg" alt="JetBrains logo.">

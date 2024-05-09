# Agents 🤖

Build Agentic workflows with function calling, powered by LangChain.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/aniketmaurya/python-project-template?template=false)

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

## Usage/Examples

### Simple tool use with a local or cloud LLM

LLM with access to weather API:

```python
from agents.llms.llm import LlamaCppChatCompletion
from agents.tools import get_current_weather, wikipedia_search

llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([get_current_weather, wikipedia_search])  # Add any tool from LangChain

messages = [
    {"role": "system", "content": "You are a helpful assistant that has access to tools and use that to help humans."},
    {"role": "user", "content": "What is the weather in London?"}
]
output = llm.chat_completion(messages)
tool_results = llm.run_tool(output)
print(tool_results)
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

You can also use Cohere's API for tool use/function calling:

```python
from agents.tools import get_current_weather

from agents.llms._cohere import CohereChatCompletion
from agents.tool_executor import need_tool_use

llm = CohereChatCompletion()
llm.bind_tools([get_current_weather])

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that has access to tools. Use the tools only if you don't have answers."
    },
    {"role": "user", "content": "how is the weather in London today?"}
]

output = llm.chat_completion(messages)

if need_tool_use(output):
    tool_results = llm.run_tool(output)
    tool_results[0]["role"] = "assistant"

    updated_messages = messages + tool_results
    updated_messages = updated_messages + [{"role": "system", "content": "Do not use any tools now."},
                                           {"role": "user",
                                            "content": "Think step by step and answer my question based on the above context."}
                                           ]
    output = llm.chat_completion(updated_messages)

print(output.choices[0].message.content)
```

<details>
    <summary>See output:</summary>

The weather in London, as of [1;92m1:58[0m PM, is partly cloudy with a temperature of [1;36m20[0m degrees Celsius or
[1;36m68[0m degrees
Fahrenheit. The humidity is at [1;36m49[0m%, and the UV index is [1;36m5[0m. There is no precipitation, and the wind
speed is mild at
[1;36m7[0m km/h or [1;36m4[0m mph. The pressure is at [1;36m1029[0m millibars, with a visibility range of [
1;36m10[0m km or [1;36m6[0m miles. So, overall, it's
a pleasant day with mild temperatures and partly cloudy skies.

</details>

### Multi-modal Agent 👁🤖️

You can build an AI Agent that can see the world with computer vision.

```python
from agents.llms.llm import LlamaCppChatCompletion
from agents.tools import get_current_weather, wikipedia_search, google_search, image_inspector

llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([google_search, wikipedia_search, image_inspector])

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
messages = [
    {"role": "system", "content": "You are a helpful assistant that has access to tools and use that to help humans."},
    {"role": "user",
     "content": f"Check this image {image_url} and suggest me a location where I can go in London which looks similar"}
]
output = llm.chat_completion(messages)
tool_results = llm.run_tool(output)

updated_messages = messages + tool_results
messages = updated_messages + [{"role": "user", "content": "please answer me, based on the tool results."}]
output = llm.chat_completion(messages)
print(output.choices[0].message.content)
```

<details>
    <summary>See output:</summary>

Based on the image you provided, it appears to be a nature boardwalk surrounded by lush green grass and a peaceful
sky filled with clouds. The presence of people indicates that it could be a popular spot for outdoor activities or
leisurely walks. If you're looking for a similar location in London, you might consider visiting one of the city's
many parks or nature reserves. Here are a few suggestions:

[1;36m1[0m. **Richmond Park**: This is the largest royal park in London and offers a variety of landscapes, including
open
grasslands, wooded areas, and lakes. It's a great place for walking, cycling, and enjoying the outdoors.

[1;36m2[0m. **Hampstead Heath**: Another large green space in London, Hampstead Heath is known for its ponds, meadows,
and
woodlands. It's a popular spot for picnics, sunbathing, and hiking.

[1;36m3[0m. **Greenwich Park**: This park offers panoramic views of the city and is home to several historic
buildings,
including the Royal Observatory. It's a great place for a leisurely walk or a picnic.

[1;36m4[0m. **Victoria Park**: A smaller but still beautiful park in East London, Victoria Park is known for its
lakes,
gardens, and outdoor events.

[1;36m5[0m. **Hyde Park**: One of the most central parks in London, Hyde Park offers a variety of attractions,
including the
Serpentine Lake, Speaker's Corner, and several monuments.

These locations all offer a peaceful and natural environment similar to the image you provided, making them
excellent choices for a day out in London.


</details>

## Acknowledgements

Built with PyCharm 🧡. Thanks to JetBrains for supporting this work by providing free credits.

<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/PyCharm_icon.svg" alt="PyCharm logo.">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jetbrains.svg" alt="JetBrains logo.">

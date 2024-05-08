import json
import requests
from agents.llms.llm import create_image_inspector
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from loguru import logger

wikipedia_api_wrapper = None
_image_inspector = None


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
      city (str): The city to fetch weather for.

    Returns:
      str: current weather condition, or None if an error occurs.
    """
    try:
        data = json.dumps(
            requests.get(f"https://wttr.in/{city}?format=j1")
            .json()
            .get("current_condition")[0]
        )
        return data
    except Exception as e:
        logger.exception(e)
        error_message = f"Error fetching current weather for {city}: {e}"
        return error_message


@tool
def wikipedia_search(query: str) -> str:
    """Tool that searches the Wikipedia API. Useful for when you need to answer
    general questions about people, places, companies, facts, historical
    events, or other subjects.

    Args:
        Input should be a search query.

    Returns:
        Wikipedia search result.
    """
    global wikipedia_api_wrapper
    if wikipedia_api_wrapper is None:
        wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1)

    return wikipedia_api_wrapper.run(query)


@tool
def google_search(query: str) -> str:
    """Performs a Google search for the given query, retrieves the top search
    result URLs and description from the page.

    Args:
        Input should be a search query.

    Returns:
        URL and description of the search result.
    """
    from googlesearch import search

    for result in search(query, num_results=1, sleep_interval=5, advanced=True):
        return f"url:{result.url},\ndescription:{result.description}"


@tool
def image_inspector(image_url_or_path: str) -> str:
    """
    Read and describe the image and objects in the image.
    Args:
        Image url or file path to image.

    Returns:
        Description of the image
    """
    global _image_inspector
    if _image_inspector is None:
        _image_inspector = create_image_inspector()

    response = _image_inspector.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url_or_path}},
                ],
            }
        ]
    )
    return response["choices"][0]["message"]["content"]

from agents.tools import *  # noqa: F403
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from typing import List

duckduckgo_search = DuckDuckGoSearchRun()
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)


def get_openai_tools() -> List[dict]:
    functions = [
        # code_interpreter,
        # google_search_and_scrape,
        # get_current_stock_price,
        # get_company_news,
        # get_company_profile,
        # get_stock_fundamentals,
        # get_financial_statements,
        # get_key_financial_ratios,
        # get_analyst_recommendations,
        # get_dividend_data,
        # get_technical_indicators,
        # duckduckgo_search,
        wikipedia
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools

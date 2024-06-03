# Serve Agents with LitServe
# https://github.com/Lightning-AI/LitServe
import litserve as ls

from agents.llms import LlamaCppChatCompletion
from agents.tools import (
    get_current_weather,
    wikipedia_search,
    google_search,
    image_inspector,
)
from agents.tool_executor import need_tool_use
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class AgentAPI(ls.LitAPI):
    def setup(self, device):
        self.llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
        self.llm.bind_tools([
            get_current_weather,
            google_search,
            wikipedia_search,
            image_inspector,
        ])

    def predict(self, messages):
        logging.debug(messages[-1])
        output = self.llm.chat_completion(messages)
        if need_tool_use(output):
            tool_results = self.llm.run_tools(output)
            updated_messages = messages + tool_results
            messages = updated_messages + [
                {
                    "role": "user",
                    "content": "please answer me, based on the tool results.",
                }
            ]
            output = self.llm.chat_completion(messages)
        yield output.choices[0].message.content


if __name__ == "__main__":
    api = AgentAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)

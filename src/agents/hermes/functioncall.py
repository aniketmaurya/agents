# Credits NousResearch
# https://github.com/NousResearch/Hermes-Function-Calling
import json

from transformers import AutoTokenizer
from llama_cpp import Llama

from . import functions
from .prompter import PromptManager
from .validator import validate_function_call_schema

from .utils import (
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls,
)


class ModelInference:
    def __init__(
        self,
        model_path="/Users/aniket/weights/llama-cpp/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf",
        chat_template="chatml",
    ):
        self.chat_template = chat_template
        self.prompter = PromptManager()
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=1,
            n_ctx=0,
            verbose=False,
            chat_format="chatml",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B", trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = get_chat_template(self.chat_template)
        inference_logger.info(self.tokenizer.special_tokens_map)

    def process_completion_and_validate(self, completion):
        assistant_message = get_assistant_message(
            completion, self.chat_template, self.tokenizer.eos_token
        )

        if assistant_message:
            validation, tool_calls, error_message = validate_and_extract_tool_calls(
                assistant_message
            )

            if validation:
                inference_logger.info(
                    f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}"
                )
                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                return tool_calls, assistant_message, error_message
        else:
            inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")

    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(functions, function_name, None)
        function_args = tool_call.get("arguments", {})

        inference_logger.info(
            f"Invoking function call {function_name} with args {function_args.values()}"
        )
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict

    def run_inference(self, prompt, tools=None) -> str:
        inputs = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
        inference_logger.info(f"inputs:\n{inputs}")
        print()
        # completions = self.model.create_chat_completion(prompt, max_tokens=2000, temperature=0.8, )
        completions = self.model(inputs, max_tokens=2000, temperature=0.8, echo=False)
        # inference_logger.info(f"completions:\n{completions}")

        completion = completions["choices"][0]["text"]
        # completion = "<tool_call>" + completions["choices"][0]["message"]["content"] + "</tool_call>"
        inference_logger.info(f"completion:\n{completion}")
        return inputs + completion

    def generate_function_call(self, query, num_fewshot, max_depth=5):
        try:
            depth = 0
            user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
            chat = [{"role": "user", "content": user_message}]
            tools = functions.get_openai_tools()
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            completion = self.run_inference(prompt, tools=tools)

            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth
                tool_calls, assistant_message, error_message = (
                    self.process_completion_and_validate(completion)
                )
                prompt.append({"role": "assistant", "content": assistant_message})

                tool_message = (
                    f"Agent iteration {depth} to assist with user query: {query}\n"
                )
                if tool_calls:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

                    for tool_call in tool_calls:
                        validation, message = validate_function_call_schema(
                            tool_call, tools
                        )
                        if validation:
                            try:
                                function_response = self.execute_function_call(
                                    tool_call
                                )
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                inference_logger.info(
                                    f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}"
                                )
                            except Exception as e:
                                inference_logger.info(
                                    f"Could not execute function: {e}"
                                )
                                raise e
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(
                            f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                        )
                        return

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                elif error_message:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(
                            f"Maximum recursion depth reached ({max_depth}). Stopping recursion."
                        )
                        return

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

            recursive_loop(prompt, completion, depth)

        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

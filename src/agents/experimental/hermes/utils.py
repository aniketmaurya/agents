# Credits NousResearch
# https://github.com/NousResearch/Hermes-Function-Calling
import ast
import os
import re
import json
import xml.etree.ElementTree as ET

from art import text2art
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_hermes_logger():
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("function-calling-inference")


def print_nous_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  nousresearch"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)
    print(ascii_art)


def get_fewshot_examples(num_fewshot):
    """Return a list of few shot examples."""
    example_path = os.path.join(script_dir, "prompt_assets", "few_shot.json")
    with open(example_path) as file:
        examples = json.load(
            file
        )  # Use json.load with the file object, not the file path
    if num_fewshot > len(examples):
        raise ValueError(
            f"Not enough examples (got {num_fewshot}, but there are only {len(examples)} examples)."
        )
    return examples[:num_fewshot]


def get_chat_template(chat_template):
    """Read chat template from jinja file."""
    template_path = os.path.join(script_dir, "chat_templates", f"{chat_template}.j2")

    if not os.path.exists(template_path):
        logging.error(f"Template file not found: {chat_template}")
        return None
    try:
        with open(template_path) as file:
            template = file.read()
        return template
    except Exception as e:
        print(f"Error loading template: {e}")
        return None


def get_assistant_message(completion, chat_template, eos_token):
    """Define and match pattern to find the assistant message."""
    completion = completion.strip()

    if chat_template == "zephyr":
        assistant_pattern = re.compile(
            r"<\|assistant\|>((?:(?!<\|assistant\|>).)*)$", re.DOTALL
        )
    elif chat_template == "chatml":
        assistant_pattern = re.compile(
            r"<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$",
            re.DOTALL,
        )

    elif chat_template == "vicuna":
        assistant_pattern = re.compile(
            r"ASSISTANT:\s*((?:(?!ASSISTANT:).)*)$", re.DOTALL
        )
    else:
        raise NotImplementedError(
            f"Handling for chat_template '{chat_template}' is not implemented."
        )

    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        assistant_content = assistant_match.group(1).strip()
        if chat_template == "vicuna":
            eos_token = f"</s>{eos_token}"
        return assistant_content.replace(eos_token, "")
    else:
        assistant_content = None
        logging.info("No match found for the assistant pattern")
        return assistant_content


def validate_and_extract_tool_calls(assistant_content):
    logging.info(f"assistant_content: {assistant_content}")
    validation_result = False
    tool_calls = []
    error_message = None

    try:
        # wrap content in root element
        xml_root_element = f"<root>{assistant_content}</root>"
        root = ET.fromstring(xml_root_element)

        # extract JSON data
        root_all = root.findall(".//tool_call")
        logging.info(f"root_all: {root_all}")
        for element in root_all:
            json_data = None
            try:
                json_text = element.text.strip()

                try:
                    # Prioritize json.loads for better error handling
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        error_message = (
                            f"JSON parsing failed with both json.loads and ast.literal_eval:\n"
                            f"- JSON Decode Error: {json_err}\n"
                            f"- Fallback Syntax/Value Error: {eval_err}\n"
                            f"- Problematic JSON text: {json_text}"
                        )
                        logging.error(error_message)
                        continue
            except Exception as e:
                error_message = f"Cannot strip text: {e}"
                logging.error(error_message)

            if json_data is not None:
                tool_calls.append(json_data)
                validation_result = True

    except ET.ParseError as err:
        error_message = f"XML Parse Error: {err}"
        logging.error(f"XML Parse Error: {err}")

    # Return default values if no valid data is extracted
    return validation_result, tool_calls, error_message


def extract_json_from_markdown(text):
    """Extracts the JSON string from the given text using a regular expression
    pattern.

    Args:
        text (str): The input text containing the JSON string.

    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    json_pattern = r"```json\r?\n(.*?)\r?\n```"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
    else:
        print("JSON string not found in the text.")
    return None

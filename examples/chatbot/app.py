import gradio as gr
from agents.llms import LlamaCppChatCompletion
from agents.tool_executor import need_tool_use
from agents.tools import (
    get_current_weather,
    wikipedia_search,
    google_search,
    image_inspector,
)

llm = LlamaCppChatCompletion.from_default_llm(n_ctx=0)
llm.bind_tools([get_current_weather, google_search, wikipedia_search, image_inspector])


def llamacpp_chat(message, history, image=None):
    history_langchain_format = [
        {
            "role": "system",
            "content": "You are a smart assistant with access to tools. Use them only if required.",
        }
    ]
    for user, ai in history:
        history_langchain_format.append({"role": "user", "content": user})
        history_langchain_format.append({"role": "assistant", "content": ai})

    if image:
        message += f"\nAttached image filepath: ```{image}```"
        history_langchain_format.append({"role": "user", "content": message})
    else:
        history_langchain_format.append({"role": "user", "content": message})
    messages = history_langchain_format

    output = llm.chat_completion(messages)
    if need_tool_use(output):
        tool_response = llm.run_tools(output)
        updated_messages = messages + tool_response
        messages = updated_messages + [
            {"role": "user", "content": "please answer me, based on the tool results."}
        ]
        output = llm.chat_completion(messages)

    return output.choices[0].message.content


gr.ChatInterface(
    llamacpp_chat,
    additional_inputs=[gr.Image(type="filepath")],
    chatbot=gr.Chatbot(height=300, likeable=True),
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7),
    title="Agent",
    description="Ask agent any question",
    theme="soft",
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).queue().launch(server_name="0.0.0.0")

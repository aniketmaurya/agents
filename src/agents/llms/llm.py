from llama_cpp.llama_tokenizer import LlamaHFTokenizer

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler


def create_image_inspector(**kwargs):
    chat_handler = MoondreamChatHandler.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="*mmproj*",
        verbose=False,
    )

    llm = Llama.from_pretrained(
        repo_id="vikhyatk/moondream2",
        filename="*text-model*",
        chat_handler=chat_handler,
        n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
        verbose=False,
        n_gpu_layers=-1,
        **kwargs,
    )
    return llm


def create_tool_use_llm(**kwargs):
    return Llama.from_pretrained(
        repo_id="meetkai/functionary-small-v2.2-GGUF",
        filename="functionary-small-v2.2.q4_0.gguf",
        chat_format="functionary-v2",
        tokenizer=LlamaHFTokenizer.from_pretrained(
            "meetkai/functionary-small-v2.2-GGUF"
        ),
        **kwargs,
    )

from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer


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

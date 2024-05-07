from agents.experimental.hermes import ModelInference
from agents.experimental.hermes.utils import get_hermes_logger

get_hermes_logger()

# model_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
model_path = "/Users/aniket/weights/llama-cpp/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"

model_inference = ModelInference(
    model_path=model_path,
)
model_inference.generate_function_call(
    "When was Golden gate bridge painted last?", 2, 5
)

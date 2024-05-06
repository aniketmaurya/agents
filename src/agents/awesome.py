from hermes.functioncall import ModelInference

model_inference = ModelInference(chat_template="chatml")
model_inference.generate_function_call(
    "What is the capital of France?", None, 2
)

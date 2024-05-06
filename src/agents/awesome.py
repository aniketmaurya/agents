from hermes.functioncall import ModelInference

model_inference = ModelInference(chat_template="chatml")
model_inference.generate_function_call(
    "When was Golden Gate bridge last painted?", None, 5
)

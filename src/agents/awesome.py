from hermes.functioncall import ModelInference

model_inference = ModelInference(chat_template="chatml")
model_inference.generate_function_call(
    "I need the current stock price of Tesla (TSLA)",
    "chatml",
    None,
    5
)

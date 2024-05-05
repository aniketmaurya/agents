from hermes.functioncall import ModelInference

model_inference = ModelInference()
model_inference.generate_function_call(
    "Who is the president of the USA? Search on google",
    "chatml",
    2,
    10
)

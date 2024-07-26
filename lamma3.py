import os
import transformers
import torch

os.environ['HF_TOKEN'] = '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'

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are helpful assistant!"},
    {"role": "user", "content": "What is the smallest country in the world?"},
]

while True:
  prompt = input("Enter a prompt: ")
  message = {"role": "user", "content": f"{ prompt }"}
  messages.append(message)
  outputs = pipeline(
    messages,
    max_new_tokens=256,
  )
  output_text = outputs[0]["generated_text"][-1]['content']
  print('Answer', output_text)


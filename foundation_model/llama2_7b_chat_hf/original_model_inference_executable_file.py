from transformers import AutoTokenizer
import transformers
import torch

model = "NousResearch/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I want to express the word love emotionally, but please recommend 2 sophisticated love expression sentences.\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
)

print("---------------------------------------------")
for seq in sequences:
    print(f"\nResult: {seq['generated_text']}\n")
    print("---------------------------------------------")

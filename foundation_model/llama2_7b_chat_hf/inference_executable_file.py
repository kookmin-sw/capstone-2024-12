import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model


def load_model(model_path):

    model_name = "NousResearch/Llama-2-7b-chat-hf"
    
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"":0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # PEFT 모델의 가중치 로드
    model = PeftModel.from_pretrained(model, model_path)

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_params)

    # 모델 평가 모드로 전환
    model.eval()

    return model, tokenizer


# 저장된 모델 경로
model_path = "/tmp/trained_model/llama2"

model = load_model(model_path)


def inference(model, tokenizer, prompt):
    # 모델과 토크나이저를 사용하여 텍스트 생성
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = "I want to express the word love emotionally, but please recommend 2 sophisticated love expression sentences."

    generated_text = inference(model, prompt)
    
    print("---------------------------------------------")
    print(f"\n{generated_text}\n")
    print("---------------------------------------------")
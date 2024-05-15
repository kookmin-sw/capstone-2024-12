import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def inference(model_path, prompt):
    # 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 원본 모델 불러오기
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"": 0}
    )

    # PEFT 구성 불러오기
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # PEFT 모델 생성
    peft_model = get_peft_model(base_model, peft_params)

    # PEFT 모델의 가중치 로드
    model = PeftModel.from_pretrained(peft_model, model_path)

    # 모델 평가 모드로 전환
    model.eval()

    # 모델과 토크나이저를 사용하여 텍스트 생성
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__manin__":
    # 저장된 모델 경로
    model_path = "/tmp/trained_model/llama2"
    prompt = "I want to express the word love emotionally, but please recommend 2 sophisticated love expression sentences."

    generated_text = inference(model_path, prompt)
    
    print("---------------------------------------------")
    print(f"\n{generated_text}\n")
    print("---------------------------------------------")
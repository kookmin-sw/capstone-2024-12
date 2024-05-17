import os
import requests
import shutil
import zipfile
import torch
from fastapi import FastAPI, Request
import uvicorn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model

app = FastAPI()

# 환경 변수에서 모델 S3 URL 가져오기
model_s3_url = os.getenv('MODEL_S3_URL')

# 모델 다운로드
model_download = requests.get(model_s3_url)
model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('temp', model_filename) 
os.makedirs('temp', exist_ok=True) 
with open(model_temp_path, 'wb') as file:
    file.write(model_download.content) 

# 기존 모델 디렉토리 삭제 및 생성
if os.path.exists('model'):
    shutil.rmtree('model')
os.makedirs('model')

# 모델 압축 해제
with zipfile.ZipFile(model_temp_path, 'r') as zip_ref:
    zip_ref.extractall('model')  

# 임시 파일 및 디렉토리 삭제
os.remove(model_temp_path) 
shutil.rmtree('temp') 

# 모델 이름 설정 (고정)
model_name = "NousResearch/Llama-2-7b-chat-hf"

# 양자화 구성 설정
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# 모델 로드 (고정된 모델 이름 사용)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)

# 로컬 경로
model_path = '/app/model/model'

# 토크나이저 로드 (로컬 경로에서 불러오기)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) 

# PEFT 모델의 가중치 로드
model = PeftModel.from_pretrained(model, model_path)

# PEFT 파라미터 설정
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

@app.get("/")
async def healthcheck():
    return {
        "body": "healthy"
    }

@app.post("/{full_path:path}")
async def inference(request: Request):
    data = await request.json()
    prompt = data.get('prompt', '')
    try:
        # 입력 데이터 준비
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    except Exception as e:
        return {
            "error": "Tokenization failed",
            "message": str(e)
        }

    with torch.no_grad():
        try:
            # 텍스트 생성
            outputs = model.generate(**inputs, max_length=1024)
        except Exception as e:
            return {
                "error": "Inference failed",
                "message": str(e)
            }
    
    try:
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return {
            "error": "Decoding failed",
            "message": str(e)
        }

    return {
        "output": generated_text
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
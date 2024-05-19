import os
import requests
import shutil
import zipfile
import torch
from fastapi import FastAPI, Request
import uvicorn
import base64
from io import BytesIO
from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin

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

def load_lora_weights(unet, text_encoder, input_dir):
    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
    LoraLoaderMixin.load_lora_into_unet(
        lora_state_dict, network_alphas=network_alphas, unet=unet
    )
    LoraLoaderMixin.load_lora_into_text_encoder(
        lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder
    )
    return unet, text_encoder

def get_pipeline(model_dir, lora_weights_dir=None):
    pipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    if lora_weights_dir:
        unet = pipeline.unet
        text_encoder = pipeline.text_encoder
        print(f"Loading LoRA weights from {lora_weights_dir}")
        unet, text_encoder = load_lora_weights(unet, text_encoder, lora_weights_dir)
        pipeline.unet = unet
        pipeline.text_encoder = text_encoder
    return pipeline

class StableDiffusionCallable:
    def __init__(self, model_dir, lora_weights_dir=None):
        print(f"Loading model from {model_dir}")
        self.pipeline = get_pipeline(model_dir, lora_weights_dir)
        self.pipeline.set_progress_bar_config(disable=True)
        if torch.cuda.is_available():
            self.pipeline.to("cuda")
        self.output_dir = "/home/ubuntu/data/generate_data"

    def __call__(self, prompt):
        for image in self.pipeline(prompt).images:
            buffered = BytesIO()
            image.save(buffered, format="JPG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
model_path = "/app/model/stable_diffusion"

model_callable = StableDiffusionCallable(model_path)

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
        img_base64 = model_callable(prompt)
    except Exception as e:
        return {
            "error": "Inference failed",
            "message": str(e)
        }

    return {
        "output": {
            "artifacts": [{
                "base64": img_base64
            }]
        }
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)

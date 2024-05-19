import os
import subprocess
import shutil
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

# 모델 파일명 및 임시 경로 설정
model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('temp', model_filename)
os.makedirs('temp', exist_ok=True)

# wget 명령어를 사용하여 모델 다운로드
subprocess.run(['wget', model_s3_url, '-O', model_temp_path], check=True)

# 기존 모델 디렉토리 삭제 및 생성
if os.path.exists('model'):
    shutil.rmtree('model')
os.makedirs('model')

# unzip 명령어를 사용하여 모델 압축 해제
subprocess.run(['unzip', model_temp_path, '-d', 'model'], check=True)

# 임시 파일 및 디렉토리 삭제
os.remove(model_temp_path)
shutil.rmtree('temp')

# 압축 해제 후 동적으로 생성된 경로 찾기
snapshots_dir = '/app/model/stable_diffusion/models--CompVis--stable-diffusion-v1-4/snapshots'
snapshot_subdirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
if not snapshot_subdirs:
    raise ValueError("No snapshot subdirectories found")
model_path = os.path.join(snapshots_dir, snapshot_subdirs[0])

print(f"Dynamic snapshot directory: {model_path}")

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
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
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
    print(prompt)
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
import os
import pickle
import base64
from fastapi import FastAPI
import torch
import uvicorn
import requests
import shutil
import zipfile
import subprocess

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

from model.model import ModelClass

try:
    device = torch.device("cuda:0")
except Exception as e:
    print(f"NVIDIA CUDA Not Found: {e}")
    os._exit(0)

try:
    model = ModelClass()
    if os.path.exists('./model/torch.pt'):
        model.load_state_dict(torch.load("./model/torch.pt"))
    model.to(device)
except Exception as e:
    print(f"Model load failed: {e}")
    os._exit(0)

app = FastAPI()

@app.get("/")
async def healthcheck():
    return {
        "body": "healthy"
    }

@app.post("/{full_path:path}")
async def inference(data: dict):
    try:
        print(data)
        decoded_data = base64.b64decode(data['body'].encode('utf-8'))
        input_data = pickle.loads(decoded_data).cuda()
    except Exception as e:
        return {
            "error": "Deserialization failed",
            "message": str(e)
        }
    with torch.no_grad():
        try:
            output = model(input_data)
        except Exception as e:
            return {
                "error": "Inference failed",
                "message": str(e)
            }
    return {
        "body": base64.b64encode(pickle.dumps(output.cpu())).decode('utf-8')
    }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
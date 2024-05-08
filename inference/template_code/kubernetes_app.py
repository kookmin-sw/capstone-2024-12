import os
import pickle
import base64
from fastapi import FastAPI
import torch
import uvicorn
import requests
import shutil
import zipfile

model_s3_url = os.getenv('MODEL_S3_URL')

model_download = requests.get(model_s3_url)
model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('temp', model_filename)
os.makedirs('temp', exist_ok=True)
with open(model_temp_path, 'wb') as file:
    file.write(model_download.content)

if os.path.exists('model'):
    shutil.rmtree('model')
os.makedirs('model')

with zipfile.ZipFile(model_temp_path, 'r') as zip_ref:
    zip_ref.extractall('model')

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
    model.load_state_dict(torch.load("./model/torch.pt"))
    model.to(device)
except Exception as e:
    print(f"Model load failed: {e}")
    os._exit(0)

app = FastAPI()

@app.post("/")
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
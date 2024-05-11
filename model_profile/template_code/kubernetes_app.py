import os
import torch
import requests
import shutil
import zipfile

MODEL_DB_URL = os.getenv('MODEL_DB_URL')

model_s3_url = os.getenv('MODEL_S3_URL')

model_download = requests.get(model_s3_url)
model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('temp', model_filename)
os.makedirs('temp', exist_ok=True)
with open(model_temp_path, 'wb') as f:
    f.write(model_download.content)

if os.path.exists('model'):
    shutil.rmtree('model')
os.makedirs('model')

with zipfile.ZipFile(model_temp_path, 'r') as zip_ref:
    zip_ref.extractall('model')

os.remove(model_temp_path)
shutil.rmtree('temp')

from model.model import ModelClass
from time import time
import psutil
import ast
import math

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
except Exception as e:
    print(f"NVIDIA CUDA Not Found: {e}")
    os._exit(1)

try:
    model = ModelClass()
    if os.path.exists('./model/torch.pt'):
        model.load_state_dict(torch.load("./model/torch.pt"))
    model.to(device)
    print(f"model : {model}")
except Exception as e:
    print(f"Model load failed: {e}")
    os._exit(1)

def inference(model, input_data):
    with torch.no_grad():
        try:
            output = model(input_data)
        except Exception as e:
            return {
                'error': 'Inference failed',
                'message': str(e)
            }
        
def get_max_used_gpu_memory(device=None):
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)

process = psutil.Process(os.getpid())
initial_memory_usage = process.memory_info().rss / 1024 ** 2
print(f"Initial RAM usage: {initial_memory_usage:.2f} GB")

if __name__ == "__main__":
    ######## DB 에서 INPUT Shape, Type, range 를 불러왔어야 함.
    value_range = ast.literal_eval("(0, 1)")
    value_type = None
    input_data = torch.randn(size=ast.literal_eval("(1, 3, 224, 224)"),
                            dtype=value_type)
    
    torch.cuda.reset_max_memory_allocated(device)
    start_time = time()
    inference(model, input_data)
    end_time = time()
    max_used_gpu_memory = get_max_used_gpu_memory(device) / 1024 ** 2
    inference_time_s = end_time - start_time

    final_memory_usage = process.memory_info().rss / 1024 ** 2

    print(f"max used cpu memory : {final_memory_usage:.2f} MB")
    print(f"max used gpu memory : {max_used_gpu_memory:.2f} MB")
    print(f"inference time : {inference_time_s:.2f} s")

    ######## DB 에 저장할 내용들
    # max_used_gpu_memory, final_memory_usage, inference_time_s
    max_used_gpu_memory = math.ceil(max_used_gpu_memory * 1.2)
    final_memory_usage = math.ceil(final_memory_usage * 1.2)
print(f"start measure API")
import os
import torch
import requests
import shutil
import zipfile
import json

MODEL_API_URL = os.getenv('MODEL_API_URL')
print(f"MODEL_API_URL : {MODEL_API_URL}")
response = requests.get(MODEL_API_URL)
if response.status_code != 200:
    print(f"Model API Response Error: {response.status_code}")
    os._exit(1)
db_data = json.loads(response.text)
print(f"DB Data : {db_data}")
try:
    model_s3_url = db_data['s3_url']
    print(f"model_s3_url : {model_s3_url}")
    model_input_shape = db_data['input_shape']
    print(f"model_input_shape : {model_input_shape}")
    model_value_type = db_data['value_type']
    print(f"model_value_type : {model_value_type}")
    model_value_range = db_data.get('value_range', None)
    print(f"model_value_range : {model_value_range}")
    model_name = db_data.get('name', None)
    print(f"model_name : {model_name}")
except Exception as e:
    print(f"Model API Response Error: {e}")
    os._exit(1)

### INITIALIZATION ###
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
### END OF INITIALIZATION ###

try:
    from model.model import ModelClass
    from time import time
    import psutil
    import ast
    import math
except Exception as e:
    print(f"Import Error: {e}")
    os._exit(1)

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
    value_type = None
    if value_type == "float16":
        value_type = torch.float16
    elif value_type == "float32":
        value_type = torch.float32
    elif value_type == "float64":
        value_type = torch.float64
    elif value_type == "int8":
        value_type = torch.int8
    elif value_type == "int16":
        value_type = torch.int16
    elif value_type == "int32":
        value_type = torch.int32
    elif value_type == "int64":
        value_type = torch.int64
    elif value_type == "bool":
        value_type = torch.bool
    if model_value_range == "":
        input_data = torch.rand(size=ast.literal_eval(model_input_shape),
                                dtype=value_type)
    else:
        value_range = ast.literal_eval(model_value_range)
        input_data = torch.randint(low=value_range[0], high=value_range[1],
                                   size=ast.literal_eval(model_input_shape))
    
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
    max_used_gpu_memory = math.ceil(max_used_gpu_memory * 1.4)
    final_memory_usage = math.ceil(final_memory_usage * 1.4)
    deploy_platform = "Serverless"

    if max_used_gpu_memory < 8:
        if inference_time_s > 3:
            deploy_platform = "nodepool-1"
    elif max_used_gpu_memory < 16:
        deploy_platform = "nodepool-2"
    elif max_used_gpu_memory < 24:
        deploy_platform = "nodepool-3"
    elif max_used_gpu_memory < 32:
        deploy_platform = "nodepool-4"
    elif max_used_gpu_memory < 80:
        deploy_platform = "nodepool-5"

    print(f"depoly platform : {deploy_platform}")
    
    will_upload_data = {
        'deploy_platform': deploy_platform,
        'max_used_ram': final_memory_usage,
        'max_used_gpu_mem': max_used_gpu_memory,
        'inference_time': inference_time_s,
    }

    response = requests.put(MODEL_API_URL, json=will_upload_data)

    if response.status_code != 200:
        print(f"DB Update Error: {response.status_code}")
        os._exit(1)
    print("DB Update Success")
    print("Response Body :", response.text)
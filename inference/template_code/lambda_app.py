import os
import sys
import pickle
import torch
import base64
import requests
import shutil
import zipfile
import json
import subprocess

model_s3_url = os.getenv('MODEL_S3_URL')

model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('/tmp', model_filename)
os.makedirs('/tmp', exist_ok=True)

subprocess.run(['wget', model_s3_url, '-O', model_temp_path], check=True)

if os.path.exists('/tmp/model'):
    shutil.rmtree('/tmp/model')
os.makedirs('/tmp/model')

subprocess.run(['unzip', model_temp_path, '-d', '/tmp/model'], check=True)

os.remove(model_temp_path)

sys.path.append('/tmp/model')
from model import ModelClass

try:
    model = ModelClass()
    model_path = '/tmp/model/torch.pt'

    if os.path.exists(model_path):
        try:
            # Try to load the model assuming it might be wrapped in a DataParallel module
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            if 'module' in state_dict:
                model.load_state_dict(state_dict['module'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e:
            # If there's any error, try loading without assuming DataParallel wrapping
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
except Exception as e:
    print(f"Model load failed: {e}")
    os._exit(0)

def handler(event, context):
    data = event['body']
    encoded_data = json.loads(data)
    try:
        decoded_data = base64.b64decode(encoded_data['body'].encode('utf-8'))
        input_data = pickle.loads(decoded_data)
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": "Deserialization failed",
                    "message": str(e)
                }
            )
        }
    with torch.no_grad():
        try:
            output = model(input_data)
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "error": "Inference failed",
                        "message": str(e)
                    }
                )
            }
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "body": base64.b64encode(pickle.dumps(output)).decode('utf-8')
            }
        )
    }
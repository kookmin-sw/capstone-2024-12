import os
import sys
import pickle
import torch
import base64
import requests
import shutil
import zipfile
import json

model_s3_url = os.getenv('MODEL_S3_URL')

model_download = requests.get(model_s3_url)
model_filename = model_s3_url.split('/')[-1]
model_temp_path = os.path.join('/tmp', model_filename)
with open(model_temp_path, 'wb') as file:
    file.write(model_download.content)

if os.path.exists('/tmp/model'):
    shutil.rmtree('/tmp/model')
os.makedirs('/tmp/model')

with zipfile.ZipFile(model_temp_path, 'r') as zip_ref:
    zip_ref.extractall('/tmp/model')

os.remove(model_temp_path)

sys.path.append('/tmp/model')
from model import ModelClass

try:
    model = ModelClass()
    model.load_state_dict(torch.load("/tmp/model/torch.pt"))
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
import os
import sys
import torch
import requests
import shutil
import json
import subprocess

eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", "ap-northeast-2",
    "--kubeconfig", kubeconfig
])

def handler(event, context):
    body = json.loads(event['body'])
    uid = body.get('uid')
    url = f"https://wpcwvjlvkl.execute-api.ap-northeast-2.amazonaws.com/sskai-api-dev/models/{uid}"
    response = requests.get(url)
    if response.status_code != 200:
        return {
            'statusCode': response.status_code,
            'errorMessage': f"{uid} 가 존재하지 않습니다."
        }
    
    
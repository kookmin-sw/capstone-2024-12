import os
import requests
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

ecr_image_url = os.environ.get('ECR_URI')
db_api_url = os.environ.get('DB_API_URL')

def create_yaml(model_api_url, uid):
    job_name = f"job-model-profile-{uid}"
    container_name = f"model-profiler-{uid}"
    container_image_name = f"job-model-profile:latest"
    container_image_url = f"{ecr_image_url}/{container_image_name}"
    nodepool_name = f"profiler-gpu-nodepool"
    content = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
spec:
  template:
    spec:
      containers:
      - name: {container_name}
        image: {container_image_url}
        imagePullPolicy: Always
        env:
        - name: MODEL_API_URL
          value: {model_api_url}
        resources:
          requests:
            memory: 13Gi
            nvidia.com/gpu: 1
          limits:
            memory: 13Gi
            nvidia.com/gpu: 1
      restartPolicy: Never
      nodeSelector:
        karpenter.sh/nodepool: {nodepool_name}
  backoffLimit: 4
    """

    filepath = f"/tmp/{job_name}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath

def apply_job_yaml(job_filename):
    result_create_job = subprocess.run([
        kubectl, "apply", "-f", job_filename, "--kubeconfig", kubeconfig
    ])
    if result_create_job.returncode != 0: print("create job returncode != 0")
    return result_create_job

def handler(event, context):
    body = json.loads(event['body'])
    try:
        uid = body['uid']
    except Exception as e:
        return {
            'statusCode': 404,
            'message': "error at get uid from requestbody",
            'errorMessage': e
        }
    model_api_url = f"{db_api_url}models/"
    model_api_url = f"{model_api_url}{uid}"
    response = requests.get(model_api_url)
    if response.status_code != 200:
        return {
            'statusCode': response.status_code,
            'errorMessage': f"{uid} 가 존재하지 않습니다."
        }
    
    try:
        yaml_filepath = create_yaml(model_api_url, uid)
    except Exception as e:
        return {
            'statusCode': 400,
            'message': "error at create yaml",
            'errorMessage': e
        }
    try:
        result = apply_job_yaml(yaml_filepath)
    except Exception as e:
        return {
            'statusCode': 400,
            'message': "error at apply job yaml",
            'errorMessage': e
        }
    
    return {
        'statusCode': 200,
        'body': "complete model profiling job"
    }
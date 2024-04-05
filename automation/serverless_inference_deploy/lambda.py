import subprocess
import os
import boto3
import json
import uuid

# terraform module 설정
local_dir = '/tmp'
os.chdir(local_dir)

subprocess.run(["mkdir","-p",".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/terraform-provider-aws_v5.43.0_x5", ".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/.terraform.lock.hcl"])

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            destination_path = os.path.join(
                local_dir, os.path.relpath(obj['Key'], s3_folder))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            s3.download_file(bucket_name, obj['Key'], destination_path)


def create_backend(username, model, hash):
# Terraform backend 생성
    terraform_backend = f"""
    terraform {{
    backend "s3" {{
        bucket = "sskai-terraform-state"
        key = "serverless_inference/{username}/{model}/{hash}/terraform.state"
        region = "ap-northeast-2"
        encrypt = true
    }}
    }}
    """

    # 파일 저장
    with open("backend.tf", "w") as f:
        f.write(terraform_backend)

def handler(event, context):

    endpoint_name = event["queryStringParameters"]["ENDPOINT_NAME"]
    type = event["queryStringParameters"]["TYPE"]
    username = event["queryStringParameters"]["USERNAME"]
    action = event["queryStringParameters"]["ACTION"]
    container_registry = event["queryStringParameters"]["CONTAINER_REGISTRY"]
    model_name = event["queryStringParameters"]["MODEL_NAME"]

    bucket_name = 'skkai-lambda-test'
    s3_folder = 'lambda-IaC_test/'

    download_s3_folder(bucket_name, s3_folder, local_dir)

    # hash 생성
    # hash = str(uuid.uuid4())
    hash = 1234

    # Terraform 바이너리의 전체 경로 설정
    terraform_binary = '/var/task/terraform'

    # backend 생성
    create_backend(username, model_name, hash)

    # Terraform init
    subprocess.run([terraform_binary, "init", "-reconfigure"])

    # Terraform apply
    if action == 'create':
        subprocess.run([terraform_binary, "apply", "-auto-approve"])
        
        # Terraform 출력값 가져오기
        output = subprocess.check_output([terraform_binary, "output", "-json"])
        outputs = json.loads(output.decode('utf-8'))

        # 'function_url' 출력값 추출
        endpoint_url = outputs['function_url']['value']

        return {
        'statusCode': 200,
        'body': f'Endpoint URL: {endpoint_url}'
        }
    
    # Terraform destroy
    if action == 'destroy':
        subprocess.run([terraform_binary, "destroy", "-auto-approve"])
        return {
        'statusCode': 200,
        'body': 'Terraform Destroyed Successfully'
        }
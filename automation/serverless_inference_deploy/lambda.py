import subprocess
import os
import json
import requests

# Terraform 바이너리의 전체 경로 설정
terraform_binary = '/var/task/terraform'

# terraform module 설정
local_dir = '/tmp'
os.chdir(local_dir)

subprocess.run(["mkdir","-p",".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/terraform-provider-aws_v5.43.0_x5", ".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/.terraform.lock.hcl"])
subprocess.run(["ln", "-s", "/var/task/main.tf", "/tmp"])

def create_backend(user_uid, type, model_uid, endpoint_name):
# Terraform backend 생성
    terraform_backend = f"""
    terraform {{
    backend "s3" {{
        bucket = "sskai-terraform-state"
        key = "{user_uid}/{model_uid}/{type}/{endpoint_name}/terraform.state"
        region = "ap-northeast-2"
        encrypt = true
    }}
    }}
    """

    # 파일 저장
    with open("backend.tf", "w") as f:
        f.write(terraform_backend)

def handler(event, context):
    params = event.get("queryStringParameters", {})
    user_uid = params.get("USER_UID")
    endpoint_name = params.get("ENDPOINT_NAME")
    model_uid = params.get("MODEL_UID")
    type = params.get("TYPE")
    action = params.get("ACTION")

    # backend 생성
    create_backend(user_uid, type, model_uid, endpoint_name)

    inference_uid = params.get("INFERENCE_UID")

    if inference_uid:        
        # Terraform destroy
        if action == 'destroy':
            subprocess.run([terraform_binary, "destroy", "-auto-approve"])
            
            delete_inference_url = "[실제 url 입력]"+inference_uid
            response = requests.delete(delete_inference_url)

            return {
            'statusCode': 200,
            'body': 'Terraform Destroyed Successfully'
            }
        
        else:
            return{
                'statusCode': 201,
                'body': 'Please check the action'
            }
        
    else:        
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

            add_inferences = {
                "user": user_uid,
                "name": endpoint_name,
                "model": model_uid,
                "type": type,
                "endpoint": endpoint_url
            }

            add_inference_url="[실제 url 입력]"
            response = requests.post(add_inference_url, json=add_inferences)

            return {
            'statusCode': 200,
            'body': f'Endpoint URL: {endpoint_url}'
            }
        
        else:
            return {
                'statusCode': 201,
                'body': 'Please check the action'
            }
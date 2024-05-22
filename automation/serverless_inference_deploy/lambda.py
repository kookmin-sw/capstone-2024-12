import subprocess
import os
import json
import requests

db_api_url = os.getenv("DB_API_URL")
container_registry = os.getenv("ECR_URI")
REGION = os.getenv("REGION")

# Terraform 바이너리의 전체 경로 설정
terraform_binary = '/var/task/terraform'

# terraform module 설정
local_dir = '/tmp'
os.chdir(local_dir)

subprocess.run(["mkdir","-p",".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/terraform-provider-aws_v5.43.0_x5", ".terraform/providers/registry.terraform.io/hashicorp/aws/5.43.0/linux_amd64"])
subprocess.run(["ln", "-s", "/var/task/.terraform.lock.hcl"])
subprocess.run(["ln", "-s", "/var/task/main.tf", "/tmp"])

def create_backend(user_uid, endpoint_uid):
# Terraform backend 생성
    bucket_name = os.getenv("STATE_BUCKET_NAME")
    terraform_backend = f"""
    terraform {{
    backend "s3" {{
        bucket = "{bucket_name}"
        key = "{user_uid}/{endpoint_uid}/terraform.state"
        region = "{REGION}"
        encrypt = true
    }}
    }}


    """

    # 파일 저장
    with open("backend.tf", "w") as f:
        f.write(terraform_backend)

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    user_uid = body.get("user")
    action = body.get("action")
    endpoint_uid = body.get("uid")
    model_s3_url = body['model']['s3_url']
    ram_size = body['model']['max_used_ram']

    # backend 생성
    create_backend(user_uid, endpoint_uid)

    # Terraform init
    subprocess.run([terraform_binary, "init", "-reconfigure"])    
    
    # Terraform apply
    if action == 'create':
        subprocess.run([terraform_binary, "apply", "--var", f"prefix={endpoint_uid}", "--var", f"container_registry={container_registry}", "--var", f"lambda_ram_size={ram_size}", "--var", f"model_s3_url={model_s3_url}", "--var", f"region={REGION}", "-auto-approve"])

        # Terraform 출력값 가져오기
        output = subprocess.check_output([terraform_binary, "output", "-json"])
        outputs = json.loads(output.decode('utf-8'))

        # 'function_url' 출력값 추출
        endpoint_url = outputs['function_url']['value']

        update_data = {
            "endpoint": endpoint_url
        }

        response = requests.put(url=f"{db_api_url}/inferences/{endpoint_uid}", json=update_data)

        return {
            'statusCode': 200,
            'body': f'Endpoint URL: {endpoint_url}'
        }
    
    # Terraform destroy
    elif action == 'delete':        
        subprocess.run([terraform_binary, "destroy", "-auto-approve"])
        
        requests.delete(url=f"{db_api_url}/inferences/{endpoint_uid}")

        return {
        'statusCode': 200,
        'body': 'Terraform Destroyed Successfully'
        }
import subprocess
import os
import boto3

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        for obj in page.get('Contents', []):
            destination_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_folder))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            s3.download_file(bucket_name, obj['Key'], destination_path)

def handler(event, context):

    bucket_name = 'skkai-lambda-test'
    s3_folder = 'tf_test/'
    local_dir = '/tmp/tf-test'

    download_s3_folder(bucket_name, s3_folder, local_dir)

    # Terraform 바이너리의 전체 경로 설정
    terraform_binary = '/var/task/terraform'
    
    # 작업 디렉토리를 /tmp로 변경
    os.chdir(local_dir)
        
    # Terraform 초기화
    subprocess.run([terraform_binary, "init"])
        
    # Terraform 적용
    subprocess.run([terraform_binary, "apply", "-auto-approve"])
        
    # Terraform 파괴
    subprocess.run([terraform_binary, "destroy", "-auto-approve"])

    return {
        'statusCode': 200,
        'body': 'Terraform Applied and Destroyed Successfully'
    }

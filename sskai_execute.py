import subprocess
import os
import re

def is_valid_region(region):
    # region 형식이 올바른지 확인 (a-b-c 형식)
    pattern = re.compile(r'^[a-z]+-[a-z]+-\d+$')
    return pattern.match(region) is not None

def is_valid_aws_profile(profile):
    # aws s3 ls 명령어로 프로파일 검증
    try:
        subprocess.run(f"aws s3 ls --profile {profile}", check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def create_setup():
    global region, awscli_profile, ecr_uri, main_suffix
    
    while True:
        region = input("Enter REGION: ")
        if is_valid_region(region):
            break
        else:
            print("Invalid REGION format. Please enter a region in the format a-b-c (e.g., ap-northeast-2).")

    while True:
        awscli_profile = input("Enter AWSCLI PROFILE: ")
        if is_valid_aws_profile(awscli_profile):
            break
        else:
            print("Invalid AWSCLI_PROFILE. Please enter a valid AWS CLI profile name.")
    
    main_suffix = input("Enter MAIN SUFFIX: ")

    # ecr_uri 생성
    account_id_command = f"aws sts get-caller-identity --query Account --output text --profile {awscli_profile}"
    account_id_result = subprocess.run(account_id_command, shell=True, capture_output=True, text=True)
    
    if account_id_result.returncode != 0:
        raise Exception(f"Failed to get account ID: {account_id_result.stderr}")
    
    account_id = account_id_result.stdout.strip()  # .strip()을 사용하여 불필요한 공백 제거
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com"

    # setup.env 파일에 저장
    with open("setup.env", "w") as f:
        f.write(f"REGION={region}\n")
        f.write(f"AWSCLI_PROFILE={awscli_profile}\n")
        f.write(f"ECR_URI={ecr_uri}\n")
        f.write(f"MAIN_SUFFIX={main_suffix}\n")

# setup.env 파일이 있는지 확인
if os.path.exists("setup.env"):
    use_existing_file = input("setup.env file exists. Do you want to use this file? (yes/no): ").strip().lower()
    if use_existing_file == 'yes':
        # setup.env 파일을 사용하여 환경 변수 설정
        with open("setup.env") as f:
            for line in f:
                if line.strip():
                    key, value = line.strip().split('=')
                    os.environ[key] = value
        region = os.environ.get('REGION')
        awscli_profile = os.environ.get('AWSCLI_PROFILE')
        ecr_uri = os.environ.get('ECR_URI')
        main_suffix = os.environ.get('MAIN_SUFFIX')
    else:
        create_setup()
else:
    create_setup()

print("0. Exit this operation.")
print("1. Build and Deploy container image.")
print("2. Deploy SSKAI infrastructure.")
while True:
    job = input("Enter the number: ").strip()
    if job == "0":
        break
    if job == "1":
        print("You can build only with x86/64 architecture and Unix kernel (Mac/Linux).\n")
        build_type = input("Enter the type of operation (create/delete): ").strip().lower()
        if build_type == "create":
            # Container build
            container_create_command = f"./container_build.sh {ecr_uri} {region} {awscli_profile}"
            print("Building and Deploying in progress.")
            print("It takes about 15 minutes.")
            subprocess.run(container_create_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Complete.")
            break
        elif build_type == "delete":
            container_delete_command = f"./delete_container.sh {ecr_uri} {region} {awscli_profile}"
            print("Deleting in progress.")
            print("It takes about 5 minutes.")
            subprocess.run(container_delete_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Complete.")
            break
        else:
            print("Invalid operation type.")
    elif job == "2":
        terraform_type = input("Enter the type of operation (create/delete): ").strip().lower()
        if terraform_type == "create":
            # Terraform init 명령 실행
            terraform_init_command = f"terraform init"
            subprocess.run(terraform_init_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Terraform apply 명령 실행
            terraform_apply_command = f"terraform apply --auto-approve --var region={region} --var awscli_profile={awscli_profile} --var container_registry={ecr_uri} --var main_suffix={main_suffix}"
            print("It takes about 20 minutes to create.")
            subprocess.run(terraform_apply_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Complete.")
            break
        elif terraform_type == "delete":
            # Terraform init 명령 실행
            terraform_init_command = f"terraform init"
            subprocess.run(terraform_init_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Terraform destroy 명령 실행
            terraform_destroy_command = f"terraform destroy --auto-approve --var region={region} --var awscli_profile={awscli_profile} --var container_registry={ecr_uri} --var main_suffix={main_suffix}"
            print("It takes about 20 minutes to delete.")            
            subprocess.run(terraform_destroy_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Complete.")
            break
        else:
            print("Invalid operation type.")
    else:
            print("Invalid operation type.")


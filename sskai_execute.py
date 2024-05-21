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
        subprocess.run(f"aws s3 ls --profile {profile}")
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
            print("Invalid REGION format. Please enter a region in the format a-b-c (e.g., us-west-2).")

    while True:
        awscli_profile = input("Enter AWSCLI_PROFILE: ")
        if is_valid_aws_profile(awscli_profile):
            break
        else:
            print("Invalid AWSCLI_PROFILE. Please enter a valid AWS CLI profile name.")
    
    main_suffix = input("Enter MAIN_SUFFIX: ")

    # ecr_uri 생성
    account_id_command = f"aws sts get-caller-identity --query Account --output text --profile {awscli_profile}"
    account_id_result = subprocess.run(account_id_command)
    account_id = account_id_result.stdout.strip()
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

type = input("Enter TYPE. (create/delete): ")

# AWS ECR 로그인 명령 실행
ecr_command = f"aws ecr get-login-password --region {region} --profile {awscli_profile} | docker login --username AWS --password-stdin {ecr_uri}"
# subprocess.run(ecr_command)

print(region)
print(awscli_profile)
print(ecr_uri)
print(main_suffix)
print(ecr_command)

if type == "create":
    # Container build
    container_command = f"./container_build.sh {region} {awscli_profile} {ecr_uri} {main_suffix}"   
    print(container_command) 
    # subprocess.run(container_command)
    # Terraform apply 명령 실행
    terraform_command = f"terraform apply --auto-approve --var main_suffix={main_suffix}"
    print(terraform_command)
    # subprocess.run(terraform_command)
elif type == "delete":
    # Terraform destroy 명령 실행
    terraform_command = f"terraform destroy --auto-approve --var main_suffix={main_suffix}"
    print(terraform_command)
    # subprocess.run(terraform_command)
    # ecr repo 삭제
    ecr_command = f""
    print(ecr_command)
    # subprocess.run(ecr_command)
else:
    print("Invalid TYPE.")

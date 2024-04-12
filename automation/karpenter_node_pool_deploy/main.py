import subprocess
import os
from nodepool_generator import *

# request 형식 (GET 요청입니다. queryStringParameter 만 존재하기 때문)
# {
#     "queryStringParameters" : {
#         "isGpu" : "true | True | false | False", GPU 패밀리 요청인지 아닌지
#     }
# }
# 이후 region 이라는 값이 추가 될 수 있습니다. 현재는 ap-northeast-2 로 고정입니다

def handler(event, context):
    params = event["queryStringParameters"]

    is_gpu = params['isGpu']
    if is_gpu is not None:
        is_gpu = is_gpu.lower() == "true"
    else:
        return {
            'statusCode': 400,
            'body': f'Unexcepted parameter value isGpu : {is_gpu}'
        }
    
    eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')
    # TODO : eks 이름이 유효한지 테스트 할 수 있는 코드

    kubectl = '/var/task/kubectl'
    kubeconfig = '/tmp/kubeconfig'

    # get eks cluster kubernetes configuration by aws cli
    result_get_kubeconfig = subprocess.run([
        "aws", "eks", "update-kubeconfig",
        "--name", eks_cluster_name,
        "--region", "ap-northeast-2",
        "--kubeconfig", kubeconfig
    ])
    if result_get_kubeconfig.returncode != 0:
        print("kubeconfig 받아오기 returncode != 0")

    if not is_gpu:
        nodepool_filename = generate_cpu_nodepool_yaml(eks_cluster_name, "ap-northeast-2")
        result_create_cpu_nodepool = subprocess.run([
            kubectl, "apply", "-f", nodepool_filename, "--kubeconfig", kubeconfig
        ])
        if result_create_cpu_nodepool != 0: print("create cpu nodepool returncode != 0")
    else:
        nodepool_filename = generate_gpu_nodepool_yaml(eks_cluster_name, "ap-northeast-2")
        result_create_gpu_nodepool = subprocess.run([
            kubectl, "apply", "-f", nodepool_filename, "--kubeconfig", kubeconfig
        ])
        if result_create_gpu_nodepool != 0: print("create gpu nodepool returncode != 0")

    return {
        'statusCode': 200,
        'body': "test complete"
    }
    

import subprocess
import os

def print_err_command(completed_process):
    print(f"error at command : {completed_process.args}")
    print(f"stdout : {completed_process.stdout}")
    print(f"stderr : {completed_process.stderr}")

def print_command(completed_process):
    print(f"stdout of {completed_process.args} : {completed_process.stdout}")
    print(f"stderr of {completed_process.args} : {completed_process.stderr}")

def handler(event, context):
    params = event["queryStringParameters"]

    command = params['command']
    if command not in ['create', 'delete']:
        return {
            'statusCode': 400,
            'body': f'Unsupported command : {command}'
        }
    eks_name = os.environ.get('EKS_CLUSTER_NAME')
    # TODO : eks 이름이 유효한지 테스트 할 수 있는 코드

    kubectl = '/var/task/kubectl'
    kubeconfig = '/tmp/kubeconfig'

    # get eks cluster kubernetes configuration by aws cli
    result_get_kubeconfig = subprocess.run([
        "aws", "eks", "update-kubeconfig",
        "--name", eks_name,
        "--region", "ap-northeast-2",
        "--kubeconfig", kubeconfig
    ])
    if result_get_kubeconfig.returncode != 0:
        print_err_command(result_get_kubeconfig)
        return {
            'statusCode': 500,
            'body': f"Internel Server Error"
        }
    print_command(result_get_kubeconfig)

    # kubectl get nodes
    result_get_nodes = subprocess.run([
        kubectl, "get", "nodes",
        "--kubeconfig", kubeconfig
    ])
    if result_get_nodes.returncode != 0:
        print_err_command(result_get_nodes)
        return {
            'statusCode': 500,
            'body': f"Internal Server Error"
        }
    print_command(result_get_nodes)

    # kubectl get pods
    result_get_pods = subprocess.run([
        kubectl, "get", "pods", "-A",
        "--kubeconfig", kubeconfig
    ])
    if result_get_pods.returncode != 0:
        print_err_command(result_get_pods)
        return {
            'statusCode': 500,
            'body': f"Internal Server Error"
        }
    print_command(result_get_pods)

    return {
        'statusCode': 200,
        'body': result_get_pods.stdout
    }
    

import subprocess
import os
from nodepool_generator import *

eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')
eks_region = os.environ.get('REGION')

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", eks_region,
    "--kubeconfig", kubeconfig
])

def apply_nodepool_yaml(eks_cluster_name, region_name, nodepool_name, nodeclass_name, family_list, capacity_type='spot'):
    nodepool_filename = generate_yaml(eks_cluster_name, region_name, nodepool_name, nodeclass_name, family_list, capacity_type)
    result_create_nodepool = subprocess.run([
        kubectl, "apply", "-f", nodepool_filename, "--kubeconfig", kubeconfig
    ])
    if result_create_nodepool.returncode != 0: print("create nodepool returncode != 0")

    return result_create_nodepool

def handler(event, context):
    ssm = boto3.client('ssm', region_name=eks_region)
    param_lambda_url = ssm.get_parameter(Name="recocommend_family_lambda_function_url", WithDecryption=False)
    recommend_lambda_url = param_lambda_url['Parameter']['Value']

    region = eks_region

    family_dict = get_instance_family(recommend_lambda_url, region)

    for nodepool_name, family_list in family_dict.items():
        nodeclass_name = 'ec2-gpu'
        result = apply_nodepool_yaml(eks_cluster_name, region, nodepool_name, nodeclass_name, family_list)

    streamlit_cpu_nodepool_name = 'streamlit-cpu-nodepool'
    streamlit_cpu_nodepool_family_list = [
        't3.nano', 't3.micro', 't3.small', 't3.medium', 't3.large', 't3.xlarge',
        'm5.large', 'm5.xlarge'
    ]
    streamlit_nodeclass_name = 'ec2-cpu'
    result = apply_nodepool_yaml(eks_cluster_name, region, streamlit_cpu_nodepool_name, streamlit_nodeclass_name, streamlit_cpu_nodepool_family_list)

    profiler_nodepool_name = 'profiler-gpu-nodepool'
    profiler_nodepool_family_list = [
        'g6.2xlarge', 'g5.2xlarge', 'g4dn.2xlarge'
    ]
    profiler_nodeclass_name = 'ec2-gpu'
    capacity_type = 'on-demand'
    result = apply_nodepool_yaml(eks_cluster_name, region, profiler_nodepool_name, profiler_nodeclass_name, profiler_nodepool_family_list, capacity_type)

    return {
        'statusCode': 200,
        'body': "complete update nodepool"
    }

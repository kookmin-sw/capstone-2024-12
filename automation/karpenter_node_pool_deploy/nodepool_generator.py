import boto3
import requests
import json
import os

def get_instance_family(lambda_url, region):
    query_params = {'region': region}
    
    # JSON으로 데이터를 인코딩하고 헤더 설정
    headers = {'Content-Type': 'application/json'}
    # POST 요청 보내기
    response = requests.post(lambda_url, json=query_params, headers=headers)
    # 응답 상태 확인
    if response.status_code != 200:
        raise Exception(f"추천 인스턴스 람다 쿼리 실패. status code: {response.status_code}")
    # 응답 데이터 처리
    data = response.json()
    return data['family']

def generate_cpu_nodepool_yaml(eks_cluster_name, region):
    ssm = boto3.client('ssm', region_name='ap-northeast-2')
    param_lambda_url = ssm.get_parameter(Name="cpu_recommend_lambda_function_url", WithDecryption=False)
    recommend_lambda_url = param_lambda_url['Parameter']['Value']
    
    family_list = get_instance_family(recommend_lambda_url, region)

    family_string = ', '.join(f'"{instance_type}"' for instance_type in family_list)

    param_role_name = ssm.get_parameter(Name="karpenter_node_role_name", WithDecryption=False)
    node_role_name = param_role_name['Parameter']['Value']

    filename = "nodepool-cpu"
    ec2_nodeclass_name = "ec2-cpu"

    content = f"""apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: {filename}
spec:
  disruption:
    consolidateAfter: 1m0s
    consolidationPolicy: WhenEmpty
    expireAfter: Never
  template:
    metadata: {{}}
    spec:
      nodeClassRef:
        name: {ec2_nodeclass_name}
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: [{family_string}]
      - key: kubernetes.io/os
        operator: In
        values: ["linux"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
      - key: karpenter.sh/capacity-type
        operator: In
        values:
        - spot
  limits:
    cpu: 100
    memory: 100Gi
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: {ec2_nodeclass_name}
spec:
  amiFamily: AL2
  role: "{node_role_name}"
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
        iops: 5000
        deleteOnTermination: true
        throughput: 1000
"""
    filepath = f"/tmp/{filename}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def generate_gpu_nodepool_yaml(eks_cluster_name, region):
    ssm = boto3.client('ssm', region_name='ap-northeast-2')
    param_lambda_url = ssm.get_parameter(Name="gpu_recommend_lambda_function_url", WithDecryption=False)
    recommend_lambda_url = param_lambda_url['Parameter']['Value']
    
    family_list = get_instance_family(recommend_lambda_url, region)

    family_string = ', '.join(f'"{instance_type}"' for instance_type in family_list)

    param_role_name = ssm.get_parameter(Name="karpenter_node_role_name", WithDecryption=False)
    node_role_name = param_role_name['Parameter']['Value']

    filename = "nodepool-gpu"
    ec2_nodeclass_name = "ec2-gpu"

    content = f"""apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: {filename}
spec:
  disruption:
    consolidateAfter: 1m0s
    consolidationPolicy: WhenEmpty
    expireAfter: Never
  template:
    metadata: {{}}
    spec:
      nodeClassRef:
        name: {ec2_nodeclass_name}
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: [{family_string}]
      - key: kubernetes.io/os
        operator: In
        values: ["linux"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
      - key: karpenter.sh/capacity-type
        operator: In
        values:
        - spot
  limits:
    cpu: 100
    memory: 100Gi
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: {ec2_nodeclass_name}
spec:
  amiFamily: AL2
  role: "{node_role_name}"
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
        iops: 5000
        deleteOnTermination: true
        throughput: 1000
"""

    filepath = f"/tmp/{filename}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath
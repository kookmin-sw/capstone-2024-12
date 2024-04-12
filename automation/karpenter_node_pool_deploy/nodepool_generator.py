import boto3
import requests
import json

def get_instance_family(lambda_url, region):
    query_params = {'region':region}
    response = requests.get(lambda_url, params=query_params)
    if response.status_code != 200:
        raise Exception(f"추천 인스턴스 람다 쿼리 실패. status code : {response.status_code}")
    data = json.loads(response.text)
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
        name: default
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
  name: default
spec:
  amiFamily: Bottlerocket
  role: "{node_role_name}"
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
"""
    with open("cpu_nodepool.yaml", 'w') as f:
        f.write(content)

    return filename

def generate_gpu_nodepool_yaml(eks_cluster_name, region):
    ssm = boto3.client('ssm', region_name='ap-northeast-2')
    param_lambda_url = ssm.get_parameter(Name="gpu_recommend_lambda_function_url", WithDecryption=False)
    recommend_lambda_url = param_lambda_url['Parameter']['Value']
    
    family_list = get_instance_family(recommend_lambda_url, region)

    family_string = ', '.join(f'"{instance_type}"' for instance_type in family_list)

    param_role_name = ssm.get_parameter(Name="karpenter_node_role_name", WithDecryption=False)
    node_role_name = param_role_name['Parameter']['Value']

    filename = "nodepool-gpu"

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
        name: default
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
      taints:
      - effect: NoSchedule
        key: nvidia.com/gpu
        value: "true"
  limits:
    cpu: 100
    memory: 100Gi
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: Bottlerocket
  role: "{node_role_name}"
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "{eks_cluster_name}"
"""
    
    # 추후에 gpu 를 사용하는 파드가 배치된다면 해당 파드 배치 정의에
    # 아래와 같은 toleration 이 포함되어야 합니다
    # 이유는 gpu nodepool 의 정의 부분의 taints 때문인데
    # 해당 taints 에서는 gpu가 필요하지 않는 Pod가
    # gpu 가 달린 노드에 배치되지 않도록 합니다.
    # tolerations:
    # - key: "nvidia.com/gpu"
    #   operator: "Equal"
    #   value: "true"
    #   effect: "NoSchedule"

    with open(f"{filename}.yaml", 'w') as f:
        f.write(content)

    return f"{filename}.yaml"
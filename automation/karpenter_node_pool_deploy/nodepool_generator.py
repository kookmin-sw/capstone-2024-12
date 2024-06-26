import boto3
import requests
import os

karpenter_node_role_parameter_name = os.environ.get('KARPENTER_NODE_ROLE_PARAMETER_NAME')

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

def generate_yaml(eks_cluster_name, region_name, nodepool_name, nodeclass_name, family_list, price_type='spot'):
    ssm = boto3.client('ssm', region_name=region_name)
    if len(family_list) == 0:
        family_list = ['t2.micro']
    family_string = ', '.join(f'"{instance_type}"' for instance_type in family_list)

    param_role_name = ssm.get_parameter(Name=karpenter_node_role_parameter_name, WithDecryption=False)
    node_role_name = param_role_name['Parameter']['Value']

    content = f"""apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: {nodepool_name}
spec:
  disruption:
    consolidateAfter: 1m0s
    consolidationPolicy: WhenEmpty
    expireAfter: Never
  template:
    metadata: {{}}
    spec:
      nodeClassRef:
        name: {nodeclass_name}
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
        - {price_type}
  limits:
    cpu: 1000
    memory: 1000Gi
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: {nodeclass_name}
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
        volumeSize: 200Gi
        volumeType: gp3
        iops: 3000
        deleteOnTermination: true
        throughput: 300
"""

    filepath = f"/tmp/{nodepool_name}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

if __name__ == "__main__":
    region = 'ap-northeast-2'
    ssm = boto3.client('ssm', region_name=region)
    param_lambda_url = ssm.get_parameter(Name="recommend_family_lambda_function_url", WithDecryption=False)
    recommend_lambda_url = param_lambda_url['Parameter']['Value']

    family_dict = get_instance_family(recommend_lambda_url, region)

    for nodepool_name, family_list in family_dict.items():
        nodeclass_name = 'ec2-gpu'
        nodepool_filename = generate_yaml('swj-eks-test', region, nodepool_name, nodeclass_name, family_list)
        print(nodepool_filename)
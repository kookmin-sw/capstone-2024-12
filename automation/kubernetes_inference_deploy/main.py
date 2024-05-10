import subprocess
import requests
import boto3
import os
import json

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'

eks_cluster_name = os.getenv('EKS_CLUSTER_NAME')
region = os.getenv("REGION")
db_api_url = os.getenv("DB_API_URL")
ecr_uri = os.getenv("ECR_URI")

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", region,
    "--kubeconfig", kubeconfig
])

def generate_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name, ram_size):
    content = f"""---
apiVersion: v1
kind: Namespace
metadata:
  name: {user_namespace}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: {user_namespace}
  name: deployment-{endpoint_uid}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: app-{endpoint_uid}
  replicas: 2
  template:
    metadata:
      labels:
        app.kubernetes.io/name: app-{endpoint_uid}
    spec:
      containers:
      - image: {ecr_uri}/kubernetes-inference:latest
        imagePullPolicy: Always
        name: app-{endpoint_uid}
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_S3_URL
          value: {model_s3_url}
        resources:
            requests:
                memory: {ram_size}M
                nvidia.com/gpu: 1
            limits:
                memory: {ram_size}M
                nvidia.com/gpu: 1
      nodeSelector:
        karpenter.sh/nodepool: {node_pool_name}
---
apiVersion: v1
kind: Service
metadata:
  namespace: {user_namespace}
  name: service-{endpoint_uid}
spec:
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
  type: ClusterIP
  selector:
    app.kubernetes.io/name: app-{endpoint_uid}
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  namespace: {user_namespace}
  name: ingress-{endpoint_uid}
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/group.name: "{user_namespace}"
spec:
  ingressClassName: alb
  rules:
    - http:
        paths:
        - path: /{endpoint_uid}
          pathType: Prefix
          backend:
            service:
              name: service-{endpoint_uid}
              port:
                number: 8080
"""

    filepath = f"/tmp/{endpoint_uid}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def apply_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name, ram_size):
    filename = generate_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name, ram_size)
    result = subprocess.run([
        kubectl, "apply", "-f", filename, "--kubeconfig", kubeconfig
    ])
    if result != 0: print("create resource returncode != 0")
    return result

def delete_resource(user_namespace, endpoint_uid):
    deployment_name = f"deployment-{endpoint_uid}"
    service_name = f"service-{endpoint_uid}"
    ingress_name = f"ingress-{endpoint_uid}"
    ingress_result = subprocess.run([
        kubectl, "-n", user_namespace, "delete",  "ingress", ingress_name, "--kubeconfig", kubeconfig
    ])
    service_result = subprocess.run([
        kubectl, "-n", user_namespace, "delete",  "service", service_name, "--kubeconfig", kubeconfig
    ])
    deployment_result = subprocess.run([
        kubectl, "-n", user_namespace, "delete",  "deployment", deployment_name, "--kubeconfig", kubeconfig
    ])
    result = 0
    if ingress_result != 0 or service_result != 0 or deployment_result != 0:
        result = 1
        print("delete resource returncode != 0")
    return result

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    # 사용자 지정 값을 어디까지 받아올 것인지?
    user_uid = body.get("user").lower()
    endpoint_uid = body.get("uid").lower()
    action = body.get("action")

    if action == "create":
        model_s3_url = body['model']['s3_url']
        node_pool_name = body['model']['deployment_type']
        ram_size = body['model']['max_used_ram']
        result = apply_yaml(user_uid, endpoint_uid, model_s3_url, node_pool_name, ram_size)
        endpoint_url = subprocess.run(f"{kubectl} get ingress -A | grep ingress-{endpoint_uid} | awk {'print $5'}", capture_output=True, text=True, shell=True).stdout.strip()
        update_data = {
            "endpiont": endpoint_url
        }
        requests.put(url=f"{db_api_url}/inferences/{endpoint_uid}", json=update_data)
        if result == 0:
            return {
                'statusCode': 200,
                'body': "complete create inference endpoint"
            }  
        else:
            return {
                'statusCode': 500,
                'body': "error with create inference endpoint"
            }
    elif action == "delete":
        result = delete_resource(user_uid, endpoint_uid)
        if result == 0:
            requests.delete(url=f"{db_api_url}/inferences/{endpoint_uid}")
            return {
                'statusCode': 200,
                'body': "complete delete inference deployment"
            }
        else:
            return {
                'statusCode': 500,
                'body': "error with delete inference endpoint"
            }
    else:
        return {
            'statusCode': 500,
            'body': "invalid action"
        }
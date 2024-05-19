import subprocess
import requests
import os
import json
import time

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

def generate_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name):
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
      - image: {ecr_uri}/diffusion-inference:latest
        imagePullPolicy: Always
        name: app-{endpoint_uid}
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_S3_URL
          value: {model_s3_url}
        resources:
            requests:
                cpu: 2000m
                memory: 7800M
                nvidia.com/gpu: 1
            limits:
                cpu: 2000m
                memory: 7800M
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

def apply_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name):
    filename = generate_yaml(user_namespace, endpoint_uid, model_s3_url, node_pool_name)
    result = subprocess.run([
        kubectl, "apply", "-f", filename, "--kubeconfig", kubeconfig
    ])
    if result.returncode != 0: print("create resource returncode != 0")
    return result.returncode

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
    if ingress_result.returncode != 0 or service_result.returncode != 0 or deployment_result.returncode != 0:
        result = 1
        print("delete resource returncode != 0")
    return result

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    user_uid = body.get("user").lower()
    endpoint_uid = body.get("uid").lower()
    action = body.get("action")

    if action == "create":
        model_s3_url = body['model']['s3_url']
        node_pool_name = "nodepool-1"
        result = apply_yaml(user_uid, endpoint_uid, model_s3_url, node_pool_name)

        cmd = "{} get ingress -A --kubeconfig {} | grep {}".format(kubectl, kubeconfig, endpoint_uid)
        time.sleep(10)
        endpoint_url = subprocess.run(cmd, capture_output=True, shell=True).stdout.decode('utf-8').strip().split()[4]
        print(f"endpoint_url: {endpoint_url}")
        update_data = {
            "endpoint": f"http://{endpoint_url}/{endpoint_uid}"
        }
        response = requests.put(url=f"{db_api_url}/inferences/{endpoint_uid}", json=update_data)
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
import subprocess
import os
import json
import requests

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'

eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')
region = os.getenv("REGION")
ecr_uri = os.getenv("ECR_URI")
db_api_url = os.getenv("DB_API_URL")

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", region,
    "--kubeconfig", kubeconfig
])

def init_streamlit(user_namespace, endpoint_uid, endpoint_url, image_name, image_py_name):
    content = f"""---
apiVersion: v1
kind: Namespace
metadata:
  name: streamlit-{user_namespace}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: streamlit-{user_namespace}
  name: deployment-streamlit-{endpoint_uid}
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: app-streamlit-{endpoint_uid}
  replicas: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: app-streamlit-{endpoint_uid}
    spec:
      containers:
      - image: {ecr_uri}/{image_name}:latest
        imagePullPolicy: Always
        command: ["streamlit"]
        args: ["run", "{image_py_name}", "--server.baseUrlPath", streamlit/{endpoint_uid}, "--server.port=8501", "--server.address=0.0.0.0"]
        name: app-streamlit-{endpoint_uid}
        ports:
        - containerPort: 8501
        env:
        - name: ENDPOINT_URL
          value: {endpoint_url}
        resources:
            requests:
            cpu: 1000M
            memory: 2048M
        limits:
            cpu: 1000M
            memory: 2048M
      nodeSelector:
        karpenter.sh/nodepool: streamlit-cpu-nodepool
---
apiVersion: v1
kind: Service
metadata:
  namespace: streamlit-{user_namespace}
  name: service-streamlit-{endpoint_uid}
spec:
  ports:
    - port: 80
      targetPort: 8501
      protocol: TCP
  type: ClusterIP
  selector:
    app.kubernetes.io/name: app-streamlit-{endpoint_uid}
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  namespace: streamlit-{user_namespace}
  name: ingress-streamlit-{endpoint_uid}
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/group.name: "streamlit-{user_namespace}"
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /streamlit/{endpoint_uid}
        pathType: Prefix
        backend:
          service:
            name: service-streamlit-{endpoint_uid}
            port:
              number: 80
"""
    filepath = f"/tmp/{endpoint_uid}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def apply_yaml(user_namespace, endpoint_uid, endpoint_url, image_name, image_py_name):
    filename = init_streamlit(user_namespace, endpoint_uid, endpoint_url, image_name, image_py_name)
    result = subprocess.run([
        kubectl, "apply", "-f", filename, "--kubeconfig", kubeconfig
    ])
    if result.returncode != 0: print("create resource returncode != 0")
    return result.returncode

def delete_resource(user_namespace, endpoint_uid):
    deployment_name = f"deployment-streamlit-{endpoint_uid}"
    service_name = f"service-streamlit-{endpoint_uid}"
    ingress_name = f"ingress-streamlit-{endpoint_uid}"
    namespace = f"streamlit-{user_namespace}"
    ingress_result = subprocess.run([
        kubectl, "-n", namespace, "delete",  "ingress", ingress_name, "--kubeconfig", kubeconfig
    ])
    service_result = subprocess.run([
        kubectl, "-n", namespace, "delete",  "service", service_name, "--kubeconfig", kubeconfig
    ])
    deployment_result = subprocess.run([
        kubectl, "-n", namespace, "delete",  "deployment", deployment_name, "--kubeconfig", kubeconfig
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
    model_type = body.get("model_type").lower()
    action = body.get("action")

    if model_type == "llama":
      image_name = "llama2-streamlit"
      image_py_name = "text-generator-llama2-13B.py"
    elif model_type == "diffusion":
      image_name = "sdxl1-streamlit"
      image_py_name = "image-generator-SDXL1.py"
    else:
        return {
            'statusCode': 400,
            'body': "Invalid Model Type"
        }

    if action == "create":
      # 추론 엔드포인트 주소
      endpoint_url = body.get("endpoint_url")
      result = apply_yaml(user_uid, endpoint_uid, endpoint_url, image_name, image_py_name)
      cmd = "{} get ingress -A --kubeconfig {} | grep {}".format(kubectl, kubeconfig, endpoint_uid)
      # streamlit endpoint 주소
      streamlit_endpoint_url = subprocess.run(cmd, capture_output=True, shell=True).stdout.decode('utf-8').strip().split()[4]
      print(f"streamlit_endpoint_url: {streamlit_endpoint_url}/streamlit/{endpoint_uid}")
      update_data = {
            "streamlit_url": f"http://{streamlit_endpoint_url}/streamlit/{endpoint_uid}"
        }
      response = requests.put(url=f"{db_api_url}/inferences/{endpoint_uid}", json=update_data)
      if result == 0:
          return {
              'statusCode': 200,
              'body': f"{streamlit_endpoint_url}/streamlit/{endpoint_uid}"
          }
      else:
          return {
              'statusCode': 500,
              'body': "error with deploy streamlit"
          }
    elif action == "delete":
        result = delete_resource(user_uid, endpoint_uid)
        update_data = {
              "streamlit_url": "-"
          }
        response = requests.put(url=f"{db_api_url}/inferences/{endpoint_uid}", json=update_data)
        if result == 0:
            return {
                'statusCode': 200,
                'body': "complete delete streamlit"
            }
        else:
            return {
                'statusCode': 500,
                'body': "error with delete streamlit"
            }
    else:
      return {
          'statusCode': 500,
          'body': "Invalid action"
      }
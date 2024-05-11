import subprocess
import os

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'
eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", "ap-northeast-2",
    "--kubeconfig", kubeconfig
])
if result_get_kubeconfig.returncode != 0:
    print("kubeconfig 받아오기 returncode != 0")


def init_rayjob(TRAIN_NAME):
    filename = "raycluster"

    content = f"""---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  namespace: kuberay
  name: rayjob-jobname
spec:
  entrypoint: python /home/ray/samples/train-code.py

  # shutdownAfterJobFinishes specifies whether the RayCluster should be deleted after the RayJob finishes. Default is false.
  shutdownAfterJobFinishes: true

  # ttlSecondsAfterFinished specifies the number of seconds after which the RayCluster will be deleted after the RayJob finishes.
  ttlSecondsAfterFinished: 10

  # RuntimeEnvYAML represents the runtime environment configuration provided as a multi-line YAML string.
  # See https://docs.ray.io/en/latest/ray-core/handling-dependencies.html for details.
  # (New in KubeRay version 1.0.)
  runtimeEnvYAML: |
    pip:
      - transformer==4.40.2
      - torch==2.3.0
      - datasets==2.19.1

    env_vars:
      counter_name: "test_counter"

  # Suspend specifies whether the RayJob controller should create a RayCluster instance.
  # If a job is applied with the suspend field set to true, the RayCluster will not be created and we will wait for the transition to false.
  # If the RayCluster is already created, it will be deleted. In the case of transition to false, a new RayCluster will be created.
  # suspend: false

  # rayClusterSpec specifies the RayCluster instance to be created by the RayJob controller.
  rayClusterSpec:
    rayVersion: '2.12.0' # should match the Ray version in the image of the containers
    # Ray head pod template
    headGroupSpec:
      # The `rayStartParams` are used to configure the `ray start` command.
      # See https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayStartParams.md for the default settings of `rayStartParams` in KubeRay.
      # See https://docs.ray.io/en/latest/cluster/cli.html#ray-start for all available options in `rayStartParams`.
      serviceType: NodePort
      rayStartParams:
        dashboard-host: '0.0.0.0'
      #pod template
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.12.0-gpu
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
              resources:
                limits:
                  cpu: "4"
                  memory: "16Gi"
                  ephemeral-storage: "50Gi"
                  nvidia.com/gpu: 1
                requests:
                  cpu: "4"
                  memory: "16Gi"
                  ephemeral-storage: "50Gi"
                  nvidia.com/gpu: 1

              volumeMounts:
                - mountPath: /home/ray/samples
                  name: train-code
          volumes:
            # You set volumes at the Pod level, then mount them into containers inside that Pod
            - name: train-code
              configMap:
                # Provide the name of the ConfigMap you want to mount.
                name: ray-job-code
                # An array of keys from the ConfigMap to create as files
                items:
                  - key: train-code.py
                    path: train-code.py
    workerGroupSpecs:
      # the pod replicas in this group typed worker
      - replicas: 1
        minReplicas: 1
        maxReplicas: 1
        # logical group name, for this called small-group, also can be functional
        groupName: small-group
        # The `rayStartParams` are used to configure the `ray start` command.
        # See https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayStartParams.md for the default settings of `rayStartParams` in KubeRay.
        # See https://docs.ray.io/en/latest/cluster/cli.html#ray-start for all available options in `rayStartParams`.
        rayStartParams: {{}}
        #pod template
        template:
          spec:
            containers:
              - name: ray-worker # must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc'
                image: rayproject/ray:2.12.0-gpu
                lifecycle:
                  preStop:
                    exec:
                      command: [ "/bin/sh","-c","ray stop" ]
                resources:
                  limits:
                    cpu: "4"
                    memory: "16Gi"
                    ephemeral-storage: "50Gi"
                    nvidia.com/gpu: 1
                  requests:
                    cpu: "4"
                    memory: "16Gi"
                    ephemeral-storage: "50Gi"
                    nvidia.com/gpu: 1

# Python Code
---
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: kuberay
  name: ray-job-code
data:
  train-code.py: |
    import ray
    import os
    import requests
    import transformer
    import torch

    ray.init()


"""
    filepath = f"/tmp/{filename}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def handler(event, context):
    train_name = event["train_name"]
    cpusize = event["cpusize"]
    gpusize = event["gpusize"]

    os.environ["TRAIN_NAME"]=train_name
    os.environ["CPU_SIZE"]=cpusize
    os.environ["GPU_SIZE"]=gpusize

    eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')
    karpenter_node_role = os.environ.get('KARPENTER_NODE_ROLE')

    
    rayjob_filename = init_rayjob(train_name)
    result_create_rayjob = subprocess.run([
            kubectl, "apply", "-f", rayjob_filename, "--kubeconfig", kubeconfig
        ])

    return {
        'statusCode': 200,
        'body': subprocess.run([kubectl, 'get', 'nodes', "--kubeconfig", kubeconfig], capture_output=True, text=True).stdout
    }
    
if __name__ == "__main__":
    print(subprocess.run(['kubectl', 'get', 'nodes'], capture_output=True, text=True).stdout)
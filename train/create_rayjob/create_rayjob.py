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

def get_requirements_txt():
    added_list = ["requests", "pendulum", "transformers", "torch", "torchvision", "datasets"]
    add_list = []
    with open("requirements.txt", "r") as f:
        for line in f:
            package_name = line.split("==")[0]
            if package_name not in added_list:
                add_list.append("      - "+line.strip()+"\n")
    return add_list

def create_yaml(TRAIN_NAME):
    filename = "rayjob"

    content = f"""---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  namespace: kuberay
  name: rayjob-jobname
spec:
  entrypoint: python /home/ray/samples/train-code.py

  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 10

  runtimeEnvYAML: |
    pip:
      - requests==2.26.0
      - pendulum==2.1.2
      - transformers==4.19.1
      - torch==2.3.0
      - torchvision==0.18.0
      - datasets==2.19.1
{''.join(get_requirements_txt())}
      
    env_vars:
      counter_name: "test_counter"

  # rayClusterSpec specifies the RayCluster instance to be created by the RayJob controller.
  rayClusterSpec:
    rayVersion: '2.12.0' # should match the Ray version in the image of the containers
    # Ray head pod template
    headGroupSpec:
      serviceType: NodePort
      rayStartParams:
        dashboard-host: '0.0.0.0'
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
                  cpu: "3500M"
                  memory: "12288M"
                  ephemeral-storage: "50Gi"
                  nvidia.com/gpu: 1
                requests:
                  cpu: "3500M"
                  memory: "12288M"
                  ephemeral-storage: "50Gi"
                  nvidia.com/gpu: 1

              volumeMounts:
                - mountPath: /home/ray/samples
                  name: train-code
          volumes:
            - name: train-code
              configMap:
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
                    cpu: "3500m"
                    memory: "12288m"
                    ephemeral-storage: "50Gi"
                    nvidia.com/gpu: 1
                  requests:
                    cpu: "3500m"
                    memory: "12288m"
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
    import torch
    import torch.nn as nn
    import torch.optim as optim

    #### added
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer, TorchConfig
    ####


    #### model

    class ModelClass(torch.nn.Module):
        def __init__(self):
            super(ModelClass, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)
        

    ####

    #### data
    # data 받아오는 아이들
    def getData():
      x = torch.randn(100, 1) * 10
      y = x + 3 * torch.randn(100,1)
      return x, y
    #### 

    #### 학습 설정들
    # 여기에서 인자 받아서 loss함수, optimizer 선택할 수 있도록.
    # def getTrainInfo(loss, optimtype, lr):
    def getTrainInfo(lr):
      model = ModelClass()
      criterion = nn.MSELoss() # "loss" 기반으로 dict에서 찾아서 지정
      optimizer = optim.SGD(model.parameters(), lr=lr)
      return model, criterion, optimizer
    ####

    #### train loop
    def train_func(epochs):
      x, y = getData()

      model, criterion, optimizer = getTrainInfo(lr=0.01)
      epochs = 100

      for epoch in range(epochs):
          model.train()
          optimizer.zero_grad()
          outputs = model(x)
          loss = criterion(outputs, y)
          loss.backward()
          optimizer.step()

          if (epoch+1) % 10 == 0:
              print(f'Epoch [{{epoch+1}}/{{epochs}}], Loss: {{loss.item():.4f}}')
      
      torch.save(model, './model.pt')

      return model

    if __name__ == "__main__":
      ray.init()

      trainer = TorchTrainer(
          train_loop_per_worker=train_func,
          train_loop_config={"lr": 0.01, "epochs": 100},
          scaling_config=train.ScalingConfig(num_workers=4, use_gpu=False)
      )

      results = trainer.fit()
      print(results)


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

    rayjob_filename = create_yaml(train_name)
    result_create_rayjob = subprocess.run([
            kubectl, "apply", "-f", rayjob_filename, "--kubeconfig", kubeconfig
        ])
    if result_create_rayjob.returncode != 0:
      print("create resource returncode != 0")
      return result_create_rayjob.returncode

    return {
        'statusCode': 200,
        'body': "Create Successfully."
    }
    
if __name__ == "__main__":
    print(subprocess.run(['kubectl', 'get', 'nodes'], capture_output=True, text=True).stdout)
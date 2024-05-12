import subprocess
import os
import json
import requests
import shutil
import zipfile

DB_API_URL = os.environ.get('DB_API_URL')
UPLOAD_S3_API_URL = os.environ.get('UPLOAD_S3_API_URL')

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

def download_and_unzip(s3_url, extract_path):
  download = requests.get(s3_url)
  filename = s3_url.split('/')[-1]
  temp_path = os.path.join('/tmp/tmp', filename)

  os.makedirs('/tmp/tmp', exist_ok=True)
  with open(temp_path, 'wb') as file:
    file.write(download.content)

  if os.path.exists(extract_path):
    shutil.rmtree(extract_path)
  os.makedirs(extract_path)

  with zipfile.ZipFile(temp_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
  
  os.remove(temp_path)
  shutil.rmtree('/tmp/tmp')

def get_requirements_txt():
  added_list = ["requests", "pendulum", "transformers", "torch", "torchvision", "datasets"]
  add_list = []
  with open("/tmp/data_load/requirements.txt", "r") as f:
    for line in f:
      package_name = line.split("==")[0]
      if package_name not in added_list:
        add_list.append("      - "+line.strip()+"\n")
  return add_list

def get_load_data_py():
  with open("/tmp/data_load/sskai_data_load.py", "r") as f:
    line_list = []
    for line in f:
      line_list.append(line.rstrip()+"\n")
  return line_list

def create_yaml(uid, user_uid, model_uid, model_s3_url, data_s3_url, data_load_s3_url, worker_num, epoch_num, optim_str, loss_str, batch_size, learning_rate, train_split_size, ram_size):
    filename = "rayjob"

    content = f"""---
apiVersion: ray.io/v1
kind: RayJob
metadata:
  namespace: kuberay
  name: rayjob-{uid}
spec:
  entrypoint: python /home/ray/train-code.py

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
                - name: train-code
                  mountPath: /home/ray/
                - name: data-load-code
                  mountPath: /home/ray/
          volumes:
            - name: train-code
              configMap:
                name: ray-job-code-{uid}
            - name: data-load-code
              configMap:
                name: ray-job-code-data-load-{uid}
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
  name: ray-job-code-{uid}
data:
  train-code.py: |
import os
import ray.train
import ray.train.torch
import requests
import shutil
import zipfile
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import ray
from ray import train
from ray.train.torch import TorchTrainer, prepare_data_loader

DB_API_URL = {DB_API_URL}
MODEL_S3_URL = {model_s3_url}
DATA_S3_URL = {data_s3_url}
DATA_LOAD_S3_URL = {data_load_s3_url}
UPLOAD_S3_API_URL = {UPLOAD_S3_API_URL}
OPTIMIZER_STR = {optim_str}
LOSS_STR = {loss_str}
LR_VALUE = {learning_rate}
EPOCH_NUM = {epoch_num}
BATCH_SIZE = {batch_size}
WORKER_NUM = {worker_num}
TRAIN_UID = {uid}
TRAIN_SPLIT_SIZE = {train_split_size}
USER_UID = {user_uid}
MODEL_UID = {model_uid}

#### 데이터 다운로드
def download_and_unzip(s3_url, extract_path):
  download = requests.get(s3_url)
  filename = s3_url.split('/')[-1]
  temp_path = os.path.join('/tmp/tmp', filename)

  os.makedirs('/tmp/tmp', exist_ok=True)
  with open(temp_path, 'wb') as file:
    file.write(download.content)

  if os.path.exists(extract_path):
    shutil.rmtree(extract_path)
  os.makedirs(extract_path)

  with zipfile.ZipFile(temp_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
  
  os.remove(temp_path)
  shutil.rmtree('/tmp/tmp')

def download_data_loader(s3_url):
  download = requests.get(s3_url)

  with open(f"./sskai_load_data.py", 'wb') as file:
    file.write(download.content)

download_and_unzip(MODEL_S3_URL, "model")
model_dir = os.getcwd() + "/model"
download_and_unzip(DATA_S3_URL, "/tmp/data")
  
from model.model import ModelClass
from sskai_load_data import sskai_load_data

#### 데이터 로딩
def load_data():
  import os
  current_dir = os.getcwd()
  os.chdir("/tmp/data")
  x, y = sskai_load_data()
  os.chdir(current_dir)
  return x, y

#### 학습 설정 함수
def getTrainInfo(optimstr, lossstr, lr):
  import torch.optim as optim
  import torch.nn as nn
  model = ModelClass()

  optimizer = eval("optim."+optimstr+"(model.parameters(), lr=lr)")
  criterion = eval("nn."+lossstr+"()")
  return model, criterion, optimizer

#### 학습 함수
def train_func(config):
  if train.get_context().get_world_rank() == 0:
    update_data = {{
      "status": "Running",
    }}
    requests.put(url=f"{{DB_API_URL}}/trains/{{TRAIN_UID}}", json=update_data)

  # 데이터 로딩
  x, y = load_data()
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TRAIN_SPLIT_SIZE)

  # 데이터셋 및 DataLoader 생성
  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
  train_loader = prepare_data_loader(train_loader)
  
  # 테스트 데이터셋
  test_dataset = TensorDataset(x_test, y_test)
  test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
  test_loader = prepare_data_loader(test_loader)

  # 모델, 손실 함수 및 옵티마이저 설정
  model, criterion, optimizer = getTrainInfo(optimstr=OPTIMIZER_STR, lossstr=LOSS_STR, lr=config["lr"])
  model = ray.train.torch.prepare_model(model)
  epochs = config["epochs"]

  # 학습 루프
  for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
      optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
      outputs = model.forward(inputs)  # 모델에 입력 데이터 전달
      loss = criterion(outputs, targets)  # 손실 계산
      loss.backward()  # 역전파 수행
      optimizer.step()  # 옵티마이저 업데이트

    model.eval()
    eval_loss = 0
    with torch.no_grad():
      for inputs, targets in test_loader:
        outputs = model(inputs)
        e_loss = criterion(outputs, targets)
        eval_loss += e_loss.item()
    eval_loss /= len(test_loader)
    
    metrics = {{"loss":loss.item(), "eval_loss":eval_loss}}
    train.report(metrics)
    
  # 최종 save 및 db 갱신
  if train.get_context().get_world_rank() == 0:
    if os.path.exists('/tmp/save_model'):
      shutil.rmtree('/tmp/save_model')
    os.makedirs('/tmp/save_model')
    shutil.copy(f'{{model_dir}}/model.py', '/tmp/save_model/model.py')
    torch.save(model.state_dict(), '/tmp/save_model/torch.pt')
    shutil.make_archive('/tmp/model', 'zip', root_dir='/tmp/save_model')

    if os.path.getsize("/tmp/model.zip")/(1024**3) < 1.5:
      with open('/tmp/model.zip', 'rb') as file:
        model_json = {{
          "upload_type": "model",
          "user_uid": USER_UID,
          "uid": MODEL_UID,
          "filename": "model.zip"
        }}
        MODEL_S3_PUT_URL = requests.post(url=UPLOAD_S3_API_URL, json=model_json).json()
        response = requests.put(url=MODEL_S3_PUT_URL["url"], data=file)

    else:
      chunk_size = 1.5 * (1024 ** 3)
      start_data_json = {{
        "upload_type": "model",
        "user_uid": USER_UID,
        "uid": MODEL_UID,
        "filename": "model.zip"
      }}
      start_multipart = requests.post(url=f"{{UPLOAD_S3_API_URL}}/start", json=start_data_json).json()
      part_list = []
      part_number = 1
      with open('/tmp/model.zip', 'rb') as file:
        while True:
          data = file.read(int(chunk_size))
          if not data:
            break
          generate_url_json = {{
            "upload_type": "model",
            "user_uid": USER_UID,
            "uid": MODEL_UID,
            "filename": "model.zip",
            "UploadId": start_multipart['UploadId'],
            "PartNumber": str(part_number)
          }}
          generate_put_url = requests.post(url=f"{{UPLOAD_S3_API_URL}}/url", json=generate_url_json).text
          response = requests.put(url=generate_put_url.strip('"'), data=data)
          part_object = {{
            "ETag": response.headers.get('etag'),
            "PartNumber": str(part_number)
          }}
          part_list.append(part_object)
          part_number += 1

        complete_url_json = {{
          "upload_type": "model",
          "user_uid": USER_UID,
          "uid": MODEL_UID,
          "filename": "model.zip",
          "UploadId": start_multipart['UploadId'],
          "Parts": part_list
        }}
        complete_url = requests.post(url=f"{{UPLOAD_S3_API_URL}}/complete", json=complete_url_json)

    update_data = {{
      "status": "Completed"
    }}
    requests.put(url=f"{{DB_API_URL}}/trains/{{TRAIN_UID}}", json=update_data)

  return model.state_dict()

if __name__ == "__main__":
  ray.init()

  trainer = TorchTrainer(
      train_loop_per_worker=train_func,
      train_loop_config={{"lr": LR_VALUE, "epochs": EPOCH_NUM, "batch_size": BATCH_SIZE}},
      scaling_config=train.ScalingConfig(num_workers=WORKER_NUM, use_gpu=False)
  )

  results = trainer.fit()
  print(results)
---
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: kuberay
  name: ray-job-code-data-load-{uid}
data:
  sskai_load_data.py: |
{''.join(get_load_data_py())}
"""
    filepath = f"/tmp/{filename}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    uid = body.get("uid")
    user_uid = body.get("user_uid")
    model_uid = body.get("model_uid")
    model_s3_url = body.get("model_s3_url")
    data_s3_url = body.get("data_s3_url")
    data_load_s3_url = body.get("data_load_s3_url")
    worker_num = body.get("worker_num")
    epoch_num = body.get("epoch_num")
    optim_str = body.get("optim_str")
    loss_str = body.get("loss_str")
    batch_size = body.get("batch_size")
    learning_rate = body.get("learning_rate")
    train_split_size = body.get("train_split_size")
    ram_size = body.get("ram_size")

    download_and_unzip(data_load_s3_url, '/tmp/data_load')

    rayjob_filename = create_yaml(uid,
                                  user_uid,
                                  model_uid,
                                  model_s3_url,
                                  data_s3_url,
                                  data_load_s3_url,
                                  worker_num,
                                  epoch_num,
                                  optim_str,
                                  loss_str,
                                  batch_size,
                                  learning_rate,
                                  train_split_size,
                                  ram_size)
    
    result_create_rayjob = subprocess.run([
            kubectl, "apply", "-f", rayjob_filename, "--kubeconfig", kubeconfig
        ])
    if result_create_rayjob.returncode != 0:
      print("create resource returncode != 0")
      return result_create_rayjob.returncode

    return {
        'statusCode': 200,
        'body': "Rayjob create successfully."
    }

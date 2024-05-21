import subprocess
import os
import json
import requests
import shutil
import zipfile

DB_API_URL = os.environ.get('DB_API_URL')
UPLOAD_S3_API_URL = os.environ.get('UPLOAD_S3_URL')
REGION = os.environ.get('REGION')
ECR_URI = os.environ.get('ECR_URI')

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'
eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", REGION,
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
  try:
    with open("/tmp/data_load/sskai_load_data.py", "r") as f:
      line_list = []
      for index, line in enumerate(f):
        if index == 0:
          line_list.append(""+line.rstrip()+"\n")
          continue
        line_list.append("        "+line.rstrip()+"\n")
    return line_list
  except FileNotFoundError:
    return []


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
      - scikit-learn==1.4.2
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
          serviceAccount: kuberay-s3-sa
          serviceAccountName: kuberay-s3-sa
          nodeSelector:
            karpenter.sh/nodepool: ray-ondemand-nodepool

          containers:
            - name: ray-head
              image: {ECR_URI}/ray-cpu:latest
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
              resources:
                limits:
                  cpu: "3500m"
                  memory: "12288M"
                  ephemeral-storage: "50Gi"
                requests:
                  cpu: "3500m"
                  memory: "12288M"
                  ephemeral-storage: "50Gi"
              volumeMounts:
                - name: train-code
                  mountPath: /home/ray/train-code.py
                  subPath: train-code.py
          volumes:
            - name: train-code
              configMap:
                name: ray-job-code-{uid}
    workerGroupSpecs:
      # the pod replicas in this group typed worker
      - replicas: {worker_num}
        minReplicas: {worker_num}
        maxReplicas: {worker_num}
        groupName: small-group
        rayStartParams: {{}}
        template:
          spec:
            serviceAccount: kuberay-s3-sa
            serviceAccountName: kuberay-s3-sa
            nodeSelector:
              karpenter.sh/nodepool: nodepool-2
            containers:
              - name: ray-worker # must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc'
                image: {ECR_URI}/ray-gpu:latest
                lifecycle:
                  preStop:
                    exec:
                      command: [ "/bin/sh","-c","ray stop" ]
                resources:
                  limits:
                    cpu: "3500m"
                    memory: "12288M"
                    ephemeral-storage: "50Gi"
                    nvidia.com/gpu: 1
                  requests:
                    cpu: "3500m"
                    memory: "12288M"
                    ephemeral-storage: "50Gi"
                    nvidia.com/gpu: 1
    
  submitterPodTemplate:
    spec:
      serviceAccount: kuberay-s3-sa
      serviceAccountName: kuberay-s3-sa
      nodeSelector:
        karpenter.sh/nodepool: ray-ondemand-nodepool
      containers:
        - name: ray-job-submitter
          image: {ECR_URI}/ray-cpu:latest
          resources:
            limits:
              cpu: "800m"
              memory: "3072M"
            requests:
              cpu: "800m"
              memory: "3072M"
      restartPolicy: Never

# Python Code
---
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: kuberay
  name: ray-job-code-{uid}
data:
  train-code.py: |
    import sys
    import re
    from urllib.parse import urlparse
    import os
    import tempfile
    import ray.train
    import ray.train.torch
    import subprocess
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
    from ray.train import RunConfig, CheckpointConfig, FailureConfig, Checkpoint, ScalingConfig

    DB_API_URL = "{DB_API_URL}"
    MODEL_S3_URL = "{model_s3_url}"
    DATA_S3_URL = "{data_s3_url}"
    DATA_LOAD_S3_URL = "{data_load_s3_url}"
    UPLOAD_S3_API_URL = "{UPLOAD_S3_API_URL}"
    OPTIMIZER_STR = "{optim_str}"
    LOSS_STR = "{loss_str}"
    LR_VALUE = {learning_rate}
    EPOCH_NUM = {epoch_num}
    BATCH_SIZE = {batch_size}
    WORKER_NUM = {worker_num}
    TRAIN_UID = "{uid}"
    TRAIN_SPLIT_SIZE = {train_split_size}
    USER_UID = "{user_uid}"
    MODEL_UID = "{model_uid}"

    def download_data_loader(s3_url):
      download = requests.get(s3_url)

      with open(f"./sskai_load_data.py", 'wb') as file:
        file.write(download.content)

    #### 데이터 로딩
    def load_data(sskai_load_data):
      import os
      current_dir = os.getcwd()
      os.chdir("/tmp/data")
      x, y = sskai_load_data()
      os.chdir(current_dir)
      return x, y

    #### 학습 설정 함수
    def getTrainInfo(ModelClass, optimstr, lossstr, lr):
      import torch.optim as optim
      import torch.nn as nn
      model = ModelClass()

      optimizer = eval("optim."+optimstr+"(model.parameters(), lr=lr)")
      criterion = eval("nn."+lossstr+"()")
      return model, criterion, optimizer

    #### 학습 함수
    def train_func(config):
      subprocess.run(['wget', '-O', '/tmp/model.zip', '{model_s3_url}'], check=True)
      subprocess.run(['unzip', '/tmp/model.zip', '-d', 'model'], check=True)
      subprocess.run(['wget', '-O', '/tmp/data.zip', '{data_s3_url}'], check=True)
      subprocess.run(['unzip', '/tmp/data.zip', '-d', '/tmp/data'], check=True)
      model_dir = os.getcwd() + "/model"
      sys.path.append(model_dir)
      sys.path.append("/tmp/data")
      
      from model import ModelClass
      {''.join(get_load_data_py())}
      # from sskai_load_data import sskai_load_data

      if train.get_context().get_world_rank() == 0:
        update_data = {{
          "status": "Running",
        }}
        requests.put(url=f"{{DB_API_URL}}/trains/{{TRAIN_UID}}", json=update_data)
      
      # 데이터 로딩
      x, y = load_data(sskai_load_data)
      if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
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
      model, criterion, optimizer = getTrainInfo(ModelClass, optimstr=OPTIMIZER_STR, lossstr=LOSS_STR, lr=config["lr"])
      model = ray.train.torch.prepare_model(model)
      if torch.cuda.is_available():
        model = model.cuda()
      epochs = config["epochs"]

      # 체크포인트 불러오기
      checkpoint = train.get_checkpoint()      

      if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
          model_state_dict = torch.load(
            os.path.join(checkpoint_dir, "model.pt")
          )
          model.load_state_dict(model_state_dict)

          optimizer_state_dict = torch.load(
            os.path.join(checkpoint_dir, "optimizer.pt")
          )
          optimizer.load_state_dict(optimizer_state_dict)

          start_epoch = torch.load(
            os.path.join(checkpoint_dir, "extra_state.pt")
          )["epoch"] + 1
          
      else:
        print("No checkpoints.")
        start_epoch = 0
      

      # 학습 루프
      for epoch in range(start_epoch, epochs):
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
        
        metrics = {{"loss":loss.item(), "eval_loss":eval_loss, "epoch":epoch}}

        # 매 에포크 checkpoint
        # 진행 에포크, 모델, optimizer 정보 
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
          checkpoint = None
          if train.get_context().get_world_rank() == 0:
            torch.save(
              model.state_dict(),  # NOTE: Unwrap the model.
              os.path.join(temp_checkpoint_dir, f"model.pt"),
            )
            torch.save(
              optimizer.state_dict(),
              os.path.join(temp_checkpoint_dir, "optimizer.pt"),
            )
            torch.save(
              {{"epoch": metrics.get("epoch")}},
              os.path.join(temp_checkpoint_dir, "extra_state.pt"),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
          train.report(metrics, checkpoint=checkpoint)

        
      # 최종 save 및 db 갱신
      if train.get_context().get_world_rank() == 0:
        if os.path.exists('/tmp/save_model'):
          shutil.rmtree('/tmp/save_model')
        os.makedirs('/tmp/save_model')
        shutil.copy(f'{{model_dir}}/model.py', '/tmp/save_model/model.py')
        model.to("cpu")
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
        
        parse_model_url = urlparse(MODEL_S3_URL)
        match_url = re.search(r'sskai-model-\w+', parse_model_url.netloc)
        model_bucket_name = match_url.group()

        update_data = {{
          "s3_url": f"https://{{model_bucket_name}}.s3.{REGION}.amazonaws.com/{{USER_UID}}/model/{{MODEL_UID}}/model.zip"
        }}
        requests.put(url=f"{{DB_API_URL}}/models/{{MODEL_UID}}", json=update_data)

    if __name__ == "__main__":
      ray.init()
      print("init done")
      parse_model_url = urlparse(MODEL_S3_URL)
      match_url = re.search(r'sskai-model-\w+', parse_model_url.netloc)
      model_bucket_name = match_url.group()
      trainer = TorchTrainer(
          train_loop_per_worker=train_func,
          train_loop_config={{"lr": LR_VALUE, "epochs": EPOCH_NUM, "batch_size": BATCH_SIZE}},
          scaling_config=train.ScalingConfig(num_workers=WORKER_NUM, use_gpu=True),
          run_config=train.RunConfig(storage_path=f"s3://{{model_bucket_name}}/{{USER_UID}}/model/{{MODEL_UID}}/",
                                    name=f"{{MODEL_UID}}",
                                    checkpoint_config=CheckpointConfig(num_to_keep=2,),
                                    failure_config=FailureConfig(max_failures=-1),
                                    )
      )

      results = trainer.fit()
      print(results)
---

"""
    filepath = f"/tmp/{filename}.yaml"
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath

def handler(event, context):
    body = json.loads(event.get("body", "{}"))
    action = body.get("action")
    uid = body.get("uid")

    if action == "create":
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
        subprocess.run([
                kubectl, "delete", "-n", "kuberay", "rayjob", f"rayjob-{uid}", "--kubeconfig", kubeconfig
        ])
        subprocess.run([
                kubectl, "delete", "-n", "kuberay", "configmap", f"ray-job-code-{uid}", "--kubeconfig", kubeconfig
        ])
        return {
          'statusCode': 500,
          'body': "Rayjob creation failure."
        }
      return {
          'statusCode': 200,
          'body': "Rayjob created successfully."
      }
    elif action == "delete":
      result_delete_rayjob = subprocess.run([
              kubectl, "delete", "-n", "kuberay", "rayjob", f"rayjob-{uid}", "--kubeconfig", kubeconfig
        ])
      
      result_delete_configmap = subprocess.run([
              kubectl, "delete", "-n", "kuberay", "configmap", f"ray-job-code-{uid}", "--kubeconfig", kubeconfig
        ])
      if ((result_delete_configmap.returncode) and (result_delete_rayjob)) != 0:
        print("delete resource returncode != 0")
        return {
          'statusCode': 500,
          'body': "Rayjob delete failure."
        }

    return {
        'statusCode': 200,
        'body': "Rayjob deleted successfully."
    }

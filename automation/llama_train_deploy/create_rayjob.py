import subprocess
import os
import json
import requests
import shutil
import zipfile

DB_API_URL = os.environ.get('DB_API_URL')
UPLOAD_S3_API_URL = os.environ.get('UPLOAD_S3_URL')
REGION = os.environ.get('REGION')
CONTAINER_REGISTRY = os.environ.get('ECR_URI')

kubectl = '/var/task/kubectl'
kubeconfig = '/tmp/kubeconfig'
eks_cluster_name = os.environ.get('EKS_CLUSTER_NAME')
region = os.environ.get('REGION')

# get eks cluster kubernetes configuration by aws cli
result_get_kubeconfig = subprocess.run([
    "aws", "eks", "update-kubeconfig",
    "--name", eks_cluster_name,
    "--region", region,
    "--kubeconfig", kubeconfig
])
if result_get_kubeconfig.returncode != 0:
    print("kubeconfig 받아오기 returncode != 0")


def create_yaml(uid, user_uid, model_uid, model_s3_url, data_s3_url, worker_num, epoch_num):
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
      - torch==2.3.0
      - datasets==2.19.1
      - transformers==4.31.0
      - bitsandbytes==0.40.2
      - accelerate==0.21.0
      - peft==0.4.0
      - pandas==2.2.2
      - pyarrow==16.0.0
      - scipy==1.13.0
      - tensorboardX==2.6.2.2
      - xformers==0.0.26.post1
      
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
              image: {CONTAINER_REGISTRY}/ray-cpu:latest
              lifecycle:
                postStart:
                  exec:
                    command: ["/bin/sh", "-c", "wget -O /tmp/data.zip {data_s3_url} && unzip /tmp/data.zip -d /tmp"]
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
                  memory: "28672M"
                  ephemeral-storage: "30Gi"
                requests:
                  cpu: "3500m"
                  memory: "28672M"
                  ephemeral-storage: "30Gi"
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
              karpenter.sh/nodepool: nodepool-3
            containers:
              - name: ray-worker
                image: {CONTAINER_REGISTRY}/ray-gpu:latest
                lifecycle:
                  postStart:
                    exec:
                      command: ["/bin/sh", "-c", "wget -O /tmp/data.zip {data_s3_url} && unzip /tmp/data.zip -d /tmp"]
                  preStop:
                    exec:
                      command: ["/bin/sh", "-c", "ray stop"]
                resources:
                  limits:
                    cpu: "3500m"
                    memory: "28672M"
                    ephemeral-storage: "30Gi"
                    nvidia.com/gpu: 1
                  requests:
                    cpu: "3500m"
                    memory: "28672M"
                    ephemeral-storage: "30Gi"
                    nvidia.com/gpu: 1
    
  submitterPodTemplate:
    spec:
      serviceAccount: kuberay-s3-sa
      serviceAccountName: kuberay-s3-sa
      nodeSelector:
        karpenter.sh/nodepool: ray-ondemand-nodepool
      containers:
        - name: ray-job-submitter
          image: rayproject/ray:2.12.0
          resources:
            limits:
              cpu: "3500m"
              memory: "12288M"
            requests:
              cpu: "3500m"
              memory: "12288M"
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
    from urllib.parse import urlparse
    import re
    import requests
    import shutil
    import tempfile
    from os import path, makedirs
    import os
    import subprocess
    import tempfile
    import torch
    from torch.utils.data import DataLoader
    from datasets import Dataset
    import pandas as pd

    import ray.train.torch as ray_torch
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, Checkpoint, FailureConfig, RunConfig, CheckpointConfig
    from ray import train

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel, LoraConfig, get_peft_model

    def create_config(epochs, model_path, data_path):
        data_size = get_datasets(data_path).num_rows
        batch_size = 1
        step = data_size // batch_size * epochs
        config = {{
            "model_path": model_path,
            "data_path" : data_path,
            "batch_size": batch_size,
            "lr": 2e-4,
            "num_epochs": epochs,
            "step": step,
            "num_workers": {worker_num},
        }}
        return config

    def get_datasets(data_path):
        dataframe = pd.read_parquet(data_path)
        dataset = Dataset.from_pandas(dataframe)
        return dataset

    def get_parquet_file_paths(directory):
        parquet_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        return parquet_files

    def load_model(model_path):
        model_name = "NousResearch/Llama-2-7b-chat-hf"
        
        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={{"":0}}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # PEFT 모델의 가중치 로드
        model = PeftModel.from_pretrained(model, model_path)

        peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_params)

        # 모델 평가 모드로 전환
        model.train()

        return model, tokenizer


    def train_func(config):
        if train.get_context().get_world_rank() == 0:
            update_data = {{
                "status": "Running",
            }}
            requests.put(url=f"{DB_API_URL}/trains/{uid}", json=update_data)    
        
        subprocess.run(['wget', '-q', '-O', '/tmp/model.zip', '{model_s3_url}'], check=True)
        subprocess.run(['unzip', '-o', '/tmp/model.zip', '-d', '/tmp'], check=True)
        
        dataset = get_datasets(config.get("data_path"))
        dataloader = DataLoader(dataset, batch_size=config.get("batch_size"), shuffle=True)
        dataloader = ray_torch.prepare_data_loader(dataloader)

        model, tokenizer = load_model(config.get("model_path"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr"), weight_decay=1e-2)

        torch.cuda.empty_cache()

        start_epoch = 0
        global_step = 0
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                optimizer.load_state_dict(
                    torch.load(path.join(checkpoint_dir, "optimizer.pt"))
                )
                start_epoch = (
                    torch.load(path.join(checkpoint_dir, "extra_state.pt"))["epoch"]
                )
                global_step = (
                    torch.load(path.join(checkpoint_dir, "extra_state.pt"))["step"]
                )
                if global_step % 1000 == 0:
                    start_epoch += 1
                model = PeftModel.from_pretrained(model, checkpoint_dir)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)


        for epoch in range(start_epoch, config.get("num_epochs")):
            for batch in dataloader:
                optimizer.zero_grad()
                inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024)
                inputs = {{k: v.cuda() for k, v in inputs.items() if k != 'token_type_ids'}}
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()

                global_step += 1

                if global_step % 500 == 0:
                    results = {{"epoch": epoch, "step": global_step, "loss": loss.item()}}
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = None
                        if not path.exists(temp_checkpoint_dir):
                            makedirs(temp_checkpoint_dir)
                        torch.save(
                            optimizer.state_dict(),
                            path.join(temp_checkpoint_dir, "optimizer.pt"),
                        )
                        torch.save(
                            {{"epoch":epoch,"step":global_step}},
                            path.join(temp_checkpoint_dir, "extra_state.pt"),
                        )
                        model.save_pretrained(temp_checkpoint_dir)
                        tokenizer.save_pretrained(temp_checkpoint_dir)

                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        train.report(results, checkpoint=checkpoint)

                if global_step >= config.get("step"):
                    break
                    
        if train.get_context().get_world_rank() == 0:
            local_save_path = "/tmp/savedmodel/savedmodel"
            if not path.exists(local_save_path):
                makedirs(local_save_path)
            torch.save(
                optimizer.state_dict(),
                path.join(local_save_path, "optimizer.pt"),
            )
            torch.save(
                {{"epoch":epoch,"step":global_step}},
                path.join(local_save_path, "extra_state.pt"),
            )
            model.save_pretrained(local_save_path)
            tokenizer.save_pretrained(local_save_path)

            print("Model informations saved to ", local_save_path)

            print("Starting to make model.zip")
            shutil.make_archive("/tmp/savedmodel", 'zip', root_dir="/tmp/savedmodel/")

            print("Zip complete. model.zip PATH = ", "/tmp/savedmodel.zip")
            

            if os.path.getsize("/tmp/savedmodel.zip")/(1024**3) < 1.5:
                with open('/tmp/savedmodel.zip', 'rb') as file:
                    model_json = {{
                    "upload_type": "model",
                    "user_uid": '{user_uid}',
                    "uid": '{model_uid}',
                    "filename": "model.zip"
                    }}
                    MODEL_S3_PUT_URL = requests.post(url='{UPLOAD_S3_API_URL}', json=model_json).json()
                    response = requests.put(url=MODEL_S3_PUT_URL["url"], data=file)

            else:
                chunk_size = 1.5 * (1024 ** 3)
                start_data_json = {{
                    "upload_type": "model",
                    "user_uid": '{user_uid}',
                    "uid": '{model_uid}',
                    "filename": "model.zip"
                }}
                start_multipart = requests.post(url=f"{UPLOAD_S3_API_URL}/start", json=start_data_json).json()
                part_list = []
                part_number = 1

            with open('/tmp/savedmodel.zip', 'rb') as file:
                while True:
                    data = file.read(int(chunk_size))
                    if not data:
                        break
                    generate_url_json = {{
                        "upload_type": "model",
                        "user_uid": '{user_uid}',
                        "uid": '{model_uid}',
                        "filename": "model.zip",
                        "UploadId": start_multipart['UploadId'],
                        "PartNumber": str(part_number)
                    }}
                    generate_put_url = requests.post(url=f"{UPLOAD_S3_API_URL}/url", json=generate_url_json).text
                    response = requests.put(url=generate_put_url.strip('"'), data=data)
                    part_object = {{
                        "ETag": response.headers.get('etag'),
                        "PartNumber": str(part_number)
                    }}
                    part_list.append(part_object)
                    part_number += 1

                complete_url_json = {{
                "upload_type": "model",
                "user_uid": '{user_uid}',
                "uid": '{model_uid}',
                "filename": "model.zip",
                "UploadId": start_multipart['UploadId'],
                "Parts": part_list
                }}
                complete_url = requests.post(url=f"{UPLOAD_S3_API_URL}/complete", json=complete_url_json)

            update_data = {{
            "status": "Completed"
            }}
            requests.put(url=f"{DB_API_URL}/trains/{uid}", json=update_data)

            parse_model_url = urlparse('{model_s3_url}')
            match_url = re.search(r'sskai-model-\w+', parse_model_url.netloc)
            model_bucket_name = match_url.group()

            update_data = {{
            "s3_url": f"https://{{model_bucket_name}}.s3.{REGION}.amazonaws.com/{user_uid}/model/{model_uid}/model.zip"
            }}
            requests.put(url=f"{DB_API_URL}/models/{model_uid}", json=update_data)

        
                    
    def run_train(config, user_id, model_id):
        # Train with Ray Train TorchTrainer.
        parse_model_url = urlparse('{model_s3_url}')
        match_url = re.search(r'sskai-model-\w+', parse_model_url.netloc)
        model_bucket_name = match_url.group()

        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=ScalingConfig(
                use_gpu=True,
                num_workers=config.get("num_workers"),
                resources_per_worker={{"GPU":1, "CPU":3}},
            ),
            run_config=RunConfig(
                storage_path=f"s3://{{model-bucket-name}}/{user_uid}/model/{model_uid}/",
                name="{model_uid}",
                checkpoint_config=CheckpointConfig(num_to_keep=2,),
                failure_config=FailureConfig(max_failures=-1),
            ),
        )
        result = trainer.fit()
        return result


    if __name__ == "__main__":
        subprocess.run(['wget', '-q', '-O', '/tmp/model.zip', '{model_s3_url}'], check=True)
        subprocess.run(['unzip', '/tmp/model.zip', '-d', '/tmp'], check=True)
        
        epochs = {epoch_num}
        model_path = "/tmp/model"

        data_path = get_parquet_file_paths("/tmp")[0]
        
        config = create_config(epochs, model_path, data_path)

        result = run_train(config, '{user_uid}', '{model_uid}')
        print(result)


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
        epoch_num = body.get("epoch")
        worker_num = 2

        rayjob_filename = create_yaml(uid, 
                                      user_uid, 
                                      model_uid, 
                                      model_s3_url, 
                                      data_s3_url, 
                                      worker_num, 
                                      epoch_num)
      
        print(rayjob_filename)

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

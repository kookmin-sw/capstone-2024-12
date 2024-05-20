import subprocess
import os
import json
import requests
import shutil
import zipfile

DB_API_URL = os.environ.get('DB_API_URL')
UPLOAD_S3_API_URL = os.environ.get('UPLOAD_S3_URL')

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



#---- 위는 람다에서.

def create_yaml(uid, user_uid, model_uid, model_s3_url, data_s3_url, worker_num, epoch_num, data_class):
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
      - diffusers==0.19.3
      - transformers==4.30.2
      - accelerate==0.20.3
      
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
              image: rayproject/ray:2.12.0
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
              resources:
                limits:
                  cpu: "800m"
                  memory: "3072M"
                  ephemeral-storage: "50Gi"
                requests:
                  cpu: "800m"
                  memory: "3072M"
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
                image: rayproject/ray:2.12.0-gpu
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
          image: rayproject/ray:2.12.0
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
    from typing import Dict

    import itertools
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        DiffusionPipeline,
        UNet2DConditionModel,
    )

    # LoRA related imports begin ##
    from diffusers.loaders import (
        LoraLoaderMixin,
        text_encoder_lora_state_dict,
    )
    from diffusers.models.attention_processor import (
        AttnAddedKVProcessor,
        AttnAddedKVProcessor2_0,
        LoRAAttnAddedKVProcessor,
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
        SlicedAttnAddedKVProcessor,
    )

    # LoRA related imports end ##
    from diffusers.utils.import_utils import is_xformers_available
    from ray.train import ScalingConfig, Checkpoint, RunConfig, FailureConfig, CheckpointConfig
    from ray import train
    from ray.train.torch import TorchTrainer

    from ray.data import read_images
    from torchvision import transforms

    import torch
    import numpy as np
    import pandas as pd
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_
    from transformers import CLIPTextModel, AutoTokenizer

    import tempfile
    from os import path, makedirs

    LORA_RANK = 4

    def get_train_dataset(config, image_resolution=512):
        # Load a directory of images as a Ray Dataset
        instance_dataset = read_images(config.get("instance_images_dir"))
        class_dataset = read_images(config.get("class_images_dir"))

        dup_times = class_dataset.count() // instance_dataset.count()
        instance_dataset = instance_dataset.map_batches(
            lambda df: pd.concat([df] * dup_times), batch_format="pandas"
        )

        # Load tokenizer for tokenizing the image prompts.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.get("model_dir"),
            subfolder="tokenizer",
        )

        def _tokenize(prompt):
            return tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.numpy()

        # Get the token ids for both prompts.
        class_prompt_ids = _tokenize(config.get("class_prompt"))[0]
        instance_prompt_ids = _tokenize(config.get("instance_prompt"))[0]

        # START: image preprocessing
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    image_resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomCrop(image_resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform_image(
            batch: Dict[str, np.ndarray], output_column_name: str
        ) -> Dict[str, np.ndarray]:
            transformed_tensors = [transform(image).numpy() for image in batch["image"]]
            batch[output_column_name] = transformed_tensors
            return batch
        # END: image preprocessing

        # START: Apply preprocessing steps as Ray Dataset operations
        # For each dataset:
        # - perform image preprocessing
        # - drop the original image column
        # - add a new column with the tokenized prompts
        instance_dataset = (
            instance_dataset.map_batches(
                transform_image, fn_kwargs={{"output_column_name": "instance_image"}}
            )
            .drop_columns(["image"])
            .add_column("instance_prompt_ids", lambda df: [instance_prompt_ids] * len(df))
        )
        # END: Apply preprocessing steps as Ray Dataset operations

        class_dataset = (
            class_dataset.map_batches(
                transform_image, fn_kwargs={{"output_column_name": "class_image"}}
            )
            .drop_columns(["image"])
            .add_column("class_prompt_ids", lambda df: [class_prompt_ids] * len(df))
        )
        # --- Ray Data

        # We may have too many duplicates of the instance images, so limit the
        # dataset size so that len(instance_dataset) == len(class_dataset)
        final_size = min(instance_dataset.count(), class_dataset.count())

        # Now, zip the images up.
        train_dataset = (
            instance_dataset.limit(final_size)
            .repartition(final_size)
            .zip(class_dataset.limit(final_size).repartition(final_size))
        )

        print("Training dataset schema after pre-processing:")
        print(train_dataset.schema())

        return train_dataset.random_shuffle()


    def collate(batch, dtype):
        images = torch.cat([batch["instance_image"], batch["class_image"]], dim=0)
        images = images.to(memory_format=torch.contiguous_format).to(dtype)

        batch_size = len(batch["instance_prompt_ids"])

        prompt_ids = torch.cat(
            [batch["instance_prompt_ids"], batch["class_prompt_ids"]], dim=0
        ).reshape(batch_size * 2, -1)

        return {{
            "images": images,
            "prompt_ids": prompt_ids,  # token ids should stay int.
        }}


    def prior_preserving_loss(model_pred, target, weight):
        # Chunk the noise and model_pred into two parts and compute
        # the loss on each part separately.
        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Compute prior loss
        prior_loss = F.mse_loss(
            model_pred_prior.float(), target_prior.float(), reduction="mean"
        )

        # Add the prior loss to the instance loss.
        return loss + weight * prior_loss

    def get_target(scheduler, noise, latents, timesteps):
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            return noise
        if pred_type == "v_prediction":
            return scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unknown prediction type {{pred_type}}")


    def add_lora_layers(unet, text_encoder):
        unet_lora_attn_procs = {{}}
        unet_lora_parameters = []
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if isinstance(
                attn_processor,
                (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0),
            ):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else LoRAAttnProcessor
                )

            module = lora_attn_processor_class(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=LORA_RANK,
            )
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())

        unet.set_attn_processor(unet_lora_attn_procs)

        text_lora_parameters = LoraLoaderMixin._modify_text_encoder(
            text_encoder, dtype=torch.float32, rank=LORA_RANK
        )

        return unet_lora_parameters, text_lora_parameters


    def load_models(config):
        # Load all models in bfloat16 to save GRAM.
        # For models that are only used for inferencing,
        # full precision is also not required.
        dtype = torch.bfloat16

        text_encoder = CLIPTextModel.from_pretrained(
            config["model_dir"],
            subfolder="text_encoder",
            torch_dtype=dtype,
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            config["model_dir"],
            subfolder="scheduler",
            torch_dtype=dtype,
        )

        # VAE is only used for inference, keeping weights in full precision is not required.
        vae = AutoencoderKL.from_pretrained(
            config["model_dir"],
            subfolder="vae",
            torch_dtype=dtype,
        )
        # We are not training VAE part of the model.
        vae.requires_grad_(False)

        # Convert unet to bf16 to save GRAM.
        unet = UNet2DConditionModel.from_pretrained(
            config["model_dir"],
            subfolder="unet",
            torch_dtype=dtype,
        )

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        if not config["use_lora"]:
            unet_trainable_parameters = unet.parameters()
            text_trainable_parameters = text_encoder.parameters()
        else:
            text_encoder.requires_grad_(False)
            unet.requires_grad_(False)
            unet_trainable_parameters, text_trainable_parameters = add_lora_layers(
                unet, text_encoder
            )

        text_encoder.train()
        unet.train()

        torch.cuda.empty_cache()

        return (
            text_encoder,
            noise_scheduler,
            vae,
            unet,
            unet_trainable_parameters,
            text_trainable_parameters,
        )


    def train_fn(config):
        start_dir = "/tmp/model"
        load_config = {{"model_dir":start_dir}}
        (
            text_encoder,
            noise_scheduler,
            vae,
            unet,
            unet_trainable_parameters,
            text_trainable_parameters,
        ) = load_models(load_config)
        optimizer.load_state_dict(
            torch.load(path.join(start_dir, "optimizer.pt"))
        )

        train_dataset = train.get_dataset_shard("train")
        # Train!
        num_train_epochs = config["num_epochs"]

        print(f"Running {{num_train_epochs}} epochs.")

        global_step = 0
        start_epoch = 0
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                load_config = {{"model_dir":checkpoint_dir}}
                (
                    text_encoder,
                    noise_scheduler,
                    vae,
                    unet,
                    unet_trainable_parameters,
                    text_trainable_parameters,
                ) = load_models(load_config)
                optimizer.load_state_dict(
                    torch.load(path.join(checkpoint_dir, "optimizer.pt"))
                )
                start_epoch = (
                    torch.load(path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
                )
                global_step = (
                    torch.load(path.join(checkpoint_dir, "extra_state.pt"))["step"]
                )

        results = {{}}
        for epoch in range(start_epoch, num_train_epochs):
            for epoch, batch in enumerate(
                train_dataset.iter_torch_batches(
                    batch_size=config["train_batch_size"],
                    device=train.torch.get_device(),
                )
            ):
                batch = collate(batch, torch.bfloat16)

                optimizer.zero_grad()

                # Convert images to latent space
                latents = vae.encode(batch["images"]).latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Predict the noise residual.
                model_pred = unet(
                    noisy_latents.to(train.torch.get_device()),
                    timesteps.to(train.torch.get_device()),
                    encoder_hidden_states.to(train.torch.get_device()),
                ).sample
                target = get_target(noise_scheduler, noise, latents, timesteps)

                loss = prior_preserving_loss(
                    model_pred, target, config["prior_loss_weight"]
                )
                loss.backward()

                # Gradient clipping before optimizer stepping.
                clip_grad_norm_(
                    itertools.chain(unet_trainable_parameters, text_trainable_parameters),
                    config["max_grad_norm"],
                )

                optimizer.step()  # Step all optimizers.

                global_step += 1
                results = {{
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.detach().item(),
                }}
                train.report(results)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if not path.exists(temp_checkpoint_dir):
                    makedirs(temp_checkpoint_dir)
                torch.save(
                    optimizer.state_dict(),
                    path.join(temp_checkpoint_dir, "optimizer.pt"),
                )
                torch.save(
                    {{"epoch":epoch, "step":global_step}},
                    path.join(temp_checkpoint_dir, "extra_state.pt"),
                )
                if not config["use_lora"]:
                    pipeline = DiffusionPipeline.from_pretrained(
                        config["model_dir"],
                        text_encoder=text_encoder,
                        unet=unet, 
                    )
                    pipeline.save_pretrained(temp_checkpoint_dir)
                else:
                    save_lora_weights(unet, text_encoder, temp_checkpoint_dir)
                
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(results, checkpoint=checkpoint)

        
        save_path = config["output_dir"]
        if train.get_context().get_world_rank() == 0:
            if not path.exists(save_path):
                makedirs(save_path)
            torch.save(
                optimizer.state_dict(),
                path.join(save_path, "optimizer.pt"),
            )
            torch.save(
                {"epoch":epoch, "step":global_step},
                path.join(save_path, "extra_state.pt"),
            )   
                    
            if not config["use_lora"]:
                pipeline = DiffusionPipeline.from_pretrained(
                    config["model_dir"],
                    text_encoder=text_encoder,
                    unet=unet,
                )
                pipeline.save_pretrained(save_path)
            else:
                save_lora_weights(unet, text_encoder, save_path)

                

            shutil.make_archive('/tmp/model', 'zip', root_dir=save_path)
            if os.path.getsize("/tmp/model.zip")/(1024**3) < 1.5:
            with open('/tmp/model.zip', 'rb') as file:
                model_json = {{
                "upload_type": "model",
                "user_uid": {user_uid},
                "uid": {model_uid},
                "filename": "model.zip"
                }}
                MODEL_S3_PUT_URL = requests.post(url={UPLOAD_S3_API_URL}, json=model_json).json()
                response = requests.put(url=MODEL_S3_PUT_URL["url"], data=file)

            else:
            chunk_size = 1.5 * (1024 ** 3)
            start_data_json = {{
                "upload_type": "model",
                "user_uid": USER_UID,
                "uid": MODEL_UID,
                "filename": "model.zip"
            }}
            start_multipart = requests.post(url=f"{UPLOAD_S3_API_URL}/start", json=start_data_json).json()
            part_list = []
            part_number = 1
            with open('/tmp/model.zip', 'rb') as file:
                while True:
                data = file.read(int(chunk_size))
                if not data:
                    break
                generate_url_json = {{
                    "upload_type": "model",
                    "user_uid": {user_uid},
                    "uid": {model_uid},
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
                "user_uid": {user_uid},
                "uid": {model_uid},
                "filename": "model.zip",
                "UploadId": start_multipart['UploadId'],
                "Parts": part_list
                }}
                complete_url = requests.post(url=f"{UPLOAD_S3_API_URL}/complete", json=complete_url_json)

            update_data = {{
            "status": "Completed"
            }}
            requests.put(url=f"{DB_API_URL}/trains/{uid}", json=update_data)
            
            update_data = {{
            "s3_url": f"https://sskai-model-storage.s3.ap-northeast-2.amazonaws.com/{user_uid}/model/{model_uid}/model.zip"
            }}
            requests.put(url=f"{DB_API_URL}/models/{uid}", json=update_data)
            
        return model.state_dict()

            


    def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
        attn_processors = unet.attn_processors

        attn_processors_state_dict = {{}}

        for attn_processor_key, attn_processor in attn_processors.items():
            for parameter_key, parameter in attn_processor.state_dict().items():
                param_name = f"{{attn_processor_key}}.{{parameter_key}}"
                attn_processors_state_dict[param_name] = parameter
        return attn_processors_state_dict


    def save_lora_weights(unet, text_encoder, output_dir):
        if not path.exists(output_dir):
            makedirs(output_dir)
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        unet_lora_layers_to_save = unet_attn_processors_state_dict(unet)
        text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder)

        LoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )


    def set_config(model_path, user_data_path, class_data_path, data_class, epoch, save_path="/tmp/model/result"):
        data_config = {{
            "model_dir":model_path,
            "instance_prompt":f"photo of the {{data_class}} that the user wants",
            "instance_images_dir":user_data_path,
            "class_prompt":f"photo of the ordinary {{data_class}}",
            "class_images_dir":class_data_path,
        }}
        train_config = {{
            "model_dir":model_path,
            "output_dir":save_path,
            "use_lora":False,
            "prior_loss_weight":1.0,
            "max_grad_norm":1.0,
            "train_batch_size":4,
            "lr":5e-6,
            "num_epochs":epoch, 
        }}
        return data_config, train_config

    def tune_model(data_config, train_config):
        # Build training datasetrain.
        train_dataset = get_train_dataset(data_config)

        print(f"Loaded training dataset (size: {{train_dataset.count()}})")
        
        # Train with Ray Train TorchTrainer.
        trainer = TorchTrainer(
            train_fn,
            train_loop_config=train_config,
            scaling_config=ScalingConfig(
                use_gpu=True,
                num_workers=4,
                resources_per_worker={{"GPU":1, "CPU":8}},
            ),
            datasets={{
                "train": train_dataset,
            }},
            run_config=train.RunConfig(storage_path=f"s3://sskai-model-storage/{user_uid}/model/{model_uid}/",
                                        name="{model_uid}",
                                        checkpoint_config=CheckpointConfig(num_to_keep=2,),
                                        failure_config=FailureConfig(max_failures=-1),
                                      ),
            failure_config=FailureConfig(max_failures=-1) # 계속 실행하게 함
        )
        result = trainer.fit()
        print(result)


    if __name__ == "__main__":
      
      # download data
      subprocess.run(['wget', '-O', '/tmp/data.zip', '{data_s3_url}'], check=True)
      subprocess.run(['unzip', '/tmp/data.zip', '-d', '/tmp/data'], check=True)
      
      # download model
      subprocess.run(['wget', '-O', '/tmp/model.zip', '{model_s3_url}', ], check=True)
      subprocess.run(['unzip', '/tmp/model.zip', '-d' ,'/tmp/model'], check=True)

      # 훈련을 위한 변수
      model_path = /tmp/model
      
      class_data_path = "/tmp/data/class_data"
      user_data_path = "/tmp/data/user_data"
      data_class = "{data_class}"
      user_epoch = {epoch_num}

      save_path="/tmp/model/result"

      data_config, train_config = set_config(model_path, user_data_path, class_data_path, data_class, user_epoch, save_path)
      tune_model(data_config, train_config)

      ls = (subprocess.run(['ls', save_path], check=True))
      print(ls.stdout)

      subprocess.run(['zip', '-r', '/tmp/model/result/model-{uid}.zip', save_path + '/*'], check=True)

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
      epoch_num = body.get("epoch_num")

      rayjob_filename = create_yaml(uid,
                                    user_uid,
                                    model_uid,
                                    model_s3_url,
                                    data_s3_url,
                                    epoch_num)
      
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



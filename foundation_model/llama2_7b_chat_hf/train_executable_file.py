import tempfile
from os import path, makedirs
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
    config = {
        "model_path": model_path,
        "data_path" : data_path,
        "batch_size": batch_size,
        "lr": 2e-4,
        "num_epochs": epochs,
        "step": step,
        "num_workers": 4,
    }
    return config


def get_datasets(data_path):
    dataframe = pd.read_parquet(data_path)
    dataset = Dataset.from_pandas(dataframe)
    return dataset


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
        device_map={"":0}
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
            inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            global_step += 1
            results = {"epoch": epoch, "step": global_step, "loss": loss.item()}

            if global_step % 100 == 0 and global_step % 500 != 0:
                train.report(results)

            if global_step % 500 == 0:
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    checkpoint = None
                    if not path.exists(temp_checkpoint_dir):
                        makedirs(temp_checkpoint_dir)
                    torch.save(
                        optimizer.state_dict(),
                        path.join(temp_checkpoint_dir, "optimizer.pt"),
                    )
                    torch.save(
                        {"epoch":epoch,"step":global_step},
                        path.join(temp_checkpoint_dir, "extra_state.pt"),
                    )
                    model.save_pretrained(temp_checkpoint_dir)
                    tokenizer.save_pretrained(temp_checkpoint_dir)

                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(results, checkpoint=checkpoint)

            if global_step >= config.get("step"):
                break
    # END: Training loop


def run_train(config, user_id, model_id):
    # Train with Ray Train TorchTrainer.
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=config.get("num_workers"),
            resources_per_worker={"GPU":1, "CPU":8},
        ),
        run_config=RunConfig(
            name=f"{model_id}",
            storage_path=f"s3://sskai-checkpoint-test/{user_id}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=2, 
            ),
            failure_config=FailureConfig(max_failures=-1) # 계속 실행하게 함
        ),
    )
    result = trainer.fit()
    return result


if __name__ == "__main__":
    epochs = 1
    model_path = "/home/ubuntu/model"
    data_path = "/home/ubuntu/data/train-00000-of-00001-9ad84bb9cf65a42f.parquet"
    
    config = create_config(epochs, model_path, data_path)

    user_id = "admin"
    model_id = "llama2"

    result = run_train(config, user_id, model_id)
    print(result)

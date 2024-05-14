import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

import ray.train.torch as ray_torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint, FailureConfig, RunConfig
from ray import train

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)


def create_config():
    epochs = 1
    batch_size = 1
    step = 1000 // batch_size * epochs
    config = {
        "model_dir":"/tmp/trained_model/llama2",
        "batch_size":batch_size,
        "lr":2e-4,
        "num_epochs":epochs,
        "step":step,
        "num_workers":1,
    }
    return config


def get_datasets():
    dataset = "mlabonne/guanaco-llama2-1k"
    dataset = load_dataset(dataset, split="train")
    return dataset


def train_func(config):
    dataset = get_datasets()
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    dataloader = ray_torch.prepare_data_loader(dataloader)

    model_name = "NousResearch/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
        # quantization_config=quant_config,
        # device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.max_position_embeddings = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)

    torch.cuda.empty_cache()

    start_epoch = 0
    global_step = 0
    gradient_clipping_value = 1.0

    os.makedirs(config["model_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["num_epochs"]):
        if global_step >= config["step"]:
            print(f"Stopping training after reaching {global_step} steps...")
            break
        for batch in dataloader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            print(inputs)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)
        
            optimizer.step()
        
            global_step += 1
            results = {"epoch":epoch, "step":global_step, "loss":loss.item()}

            if global_step % 100 == 0:
                train.report(results)

            if global_step % 500 == 0:
                torch.save(
                    model.state_dict(), 
                    os.path.join(config["model_dir"], "model_state_dict.pt")
                )
                tokenizer.save_pretrained(config["model_dir"])
            if global_step >= config["step"]:
                 break
    
        
def run_train(config, user_id, model_id):
        # Train with Ray Train TorchTrainer.
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=config["num_workers"],
        ),
        run_config=RunConfig(
            name=f"{model_id}", # user의 model name 이 들어가야 함
            storage_path=f"s3://sskai-checkpoint-test/{user_id}", # "s3://{bucket_name}/{user_name}
            # checkpoint_config=CheckpointConfig(
            #     num_to_keep=1, # 가장 마지막으로 저장된 체크포인트만을 s3에 저장함.
            # ),
            failure_config=FailureConfig(max_failures=-1) # 계속 실행하게 함
        ),
    )
    result = trainer.fit()
    return result


if __name__ == "__main__":
    config = create_config()

    result = run_train(config, "admin", "llama2")
    print(result)
import os, tempfile
import torch
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
)
from peft import PeftModel, LoraConfig, get_peft_model


def create_config():
    epochs = 1
    batch_size = 1
    step = 1000 // batch_size * epochs
    config = {
        "batch_size": batch_size,
        "lr": 2e-4,
        "num_epochs": epochs,
        "step": step,
        "num_workers": 1,
    }
    return config


def get_datasets():
    dataset = "mlabonne/guanaco-llama2-1k"
    dataset = load_dataset(dataset, split="train")
    return dataset


def train_func(config):
    dataset = get_datasets()
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size"), shuffle=True)
    dataloader = ray_torch.prepare_data_loader(dataloader)

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model_name = "NousResearch/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"":0}
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.max_position_embeddings = 1024

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr"), weight_decay=1e-2)

    torch.cuda.empty_cache()

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_params)

    start_epoch = 0
    global_step = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
            )
            start_epoch = (
                torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"]
            )
            global_step = (
                torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["step"]
            )
            if global_step % 1000 == 0:
                start_epoch += 1
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)


    for epoch in range(start_epoch, config.get("num_epochs")):
        model.train()
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
                    if not os.path.exists(temp_checkpoint_dir):
                        os.makedirs(temp_checkpoint_dir)
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(temp_checkpoint_dir, "optimizer.pt"),
                    )
                    torch.save(
                        {"epoch":epoch,"step":global_step},
                        os.path.join(temp_checkpoint_dir, "extra_state.pt"),
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
        ),
        run_config=RunConfig(
            name=f"{model_id}",
            storage_path=f"s3://sskai-checkpoint-test/{user_id}",
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

    user_id = "admin"
    model_id = "llama2"

    result = run_train(config, user_id, model_id)
    print(result)

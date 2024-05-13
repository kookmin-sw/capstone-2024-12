import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset

import ray.train.torch as ray_torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
# from peft import LoraConfig
# from trl import SFTTrainer


def create_config():
    epochs = 1
    step = 500 * epochs
    config = {
        "model_dir":"/tmp/trained_model/llama2",
        "batch_size":2,
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
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    start_epoch = 0
    for epoch in range(start_epoch, config["num_epochs"]):  
        pass                                             
        # outputs = model(inputs, labels=labels)                                                                            
        # loss = outputs.loss                                                                                               
                                                                                                                        
        # optimizer.zero_grad()                                                                                             
        # loss.backward()                                                                                                   
        # optimizer.step()   


if __name__ == "__main__":
    config = create_config()
    # Train with Ray Train TorchTrainer.
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=config["num_workers"],
        ),
    )
    result = trainer.fit()
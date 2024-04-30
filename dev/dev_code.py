import torch, os, tempfile
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ray.train.torch as ray_torch
from ray import train
from ray.train import Checkpoint, ScalingConfig
import ray
import time

import sskai_checkpoint

ray.init()

def get_datasets():
    return datasets.FashionMNIST(
        root="/tmp/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits
    

user_id = "user-1234-5678"
model_name = "test_model"
s3_path = "db_stop_checkpoint_path" # 처음에는 None 값이길 바람. 다시 실행할 때 쓰는 용도이기 때문에.

run_config = sskai_checkpoint.create_runconfig(user_id, model_name)


def train_func(config):
    start_time = time.time()
    batch_size = 64

    dataset = get_datasets()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = ray_torch.prepare_data_loader(dataloader)

    model = NeuralNetwork()
    model = ray_torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    # 이 부분부터 모듈로 넘기고 싶었는데 안되서 우선 올립니다.
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                # map_location=...,  # Load onto a different device if needed.
            )
            model.module.load_state_dict(model_state_dict)
            optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
            )
            start_epoch = (
                torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
            )


    for epoch in range(start_epoch, config["num_epochs"]):
        if train.get_context().get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                pred = model(inputs)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            # print(f"epoch: {epoch}, loss: {loss.item()}")



        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if train.get_context().get_world_rank() == 0:
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, f"model.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(temp_checkpoint_dir, "optimizer.pt"),
                )
                torch.save(
                    {"epoch": epoch},
                    os.path.join(temp_checkpoint_dir, "extra_state.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            train.report({"epoch":epoch, "loss": loss.item()}, checkpoint=checkpoint)

        if epoch == 1:
            raise RuntimeError("Intentional error to showcase restoration!")


if s3_path == None:
    trainer = ray_torch.TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"num_epochs": 5},
        scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
        run_config=run_config,
    )
    result = trainer.fit()
    s3_path = result.checkpoint # 사실상 db에 result.checkpoint 저장이 필요함.
    
else:
    restored_trainer = ray_torch.TorchTrainer.restore(
        path=os.path.expanduser(s3_path),
        train_loop_per_worker=train_func,
        train_loop_config={"num_epochs": 5},
    )
    restored_result = restored_trainer.fit()


print("------------------------------------------------------")
print(f"{result}")
print(f"best checkpoint : {result.best_checkpoints}")
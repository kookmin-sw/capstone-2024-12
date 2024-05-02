import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ray.train.torch as ray_torch
from ray import train
import ray, pyarrow
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

run_config = sskai_checkpoint.create_runconfig(user_id, model_name)

s3_path = None
# s3_path = Checkpoint(
#     path=f"{run_config.storage_path}/{run_config.name}", 
#     filesystem = pyarrow.fs.S3FileSystem(),
# )


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


    start_epoch, model, optimizer = sskai_checkpoint.load_train_logs(model, optimizer) # model과 optimizer를 로드해와야 하기 때문에 반드시 제대로 넣어주어야 함


    # range(start_epoch, config["num_epochs"])를 유지해야 자동화 포맷 유지 가능
    for epoch in range(start_epoch, config["num_epochs"]):
        if train.get_context().get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                pred = model(inputs)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

            # 저장하고 싶은 성능지표는 반드시 dic type으로 저장
            matrixs = {"epoch":epoch, "loss": loss.item()}
            sskai_checkpoint.save_train_report(matrixs, model, optimizer)

        if epoch == 1:
            raise RuntimeError("Intentional error to showcase restoration!")


result, best_result = sskai_checkpoint.run_train(s3_path, train_func, 5, 4, run_config, "loss")


print("------------------------------------------------------")
print(f"{result}")
print(f"best checkpoint : {best_result}")
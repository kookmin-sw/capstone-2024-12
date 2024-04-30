from ray.train import RunConfig, CheckpointConfig, FailureConfig, Checkpoint
import os, tempfile, torch
from ray import train


def get_checkpoint_logs(model, optimizer):
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
    return start_epoch


def create_runconfig(user_id, model_name):
    run_config = RunConfig(
        name=f"{model_name}", # user의 model name 이 들어가야 함
        storage_path=f"s3://sskai-checkpoint-test/{user_id}", # "s3://{bucket_name}/{user_name}
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=1, # 가장 마지막으로 저장된 체크포인트만을 s3에 저장함.
        # ),
        failure_config=FailureConfig(max_failures=-1) # 계속 실행하게 함
    )
    return run_config

if __name__ == "__main__":
    config = create_runconfig("user", "model")
    print(f"{config.storage_path}/{config.name}")
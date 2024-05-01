from ray.train import RunConfig, CheckpointConfig, FailureConfig, Checkpoint, ScalingConfig
import os, tempfile, torch
import ray.train.torch as ray_torch
from ray import train


# Run Config 제작하기
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

# 원격저장소에서 체크포인트 불러오기
def load_train_logs(model, optimizer):
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
    return start_epoch, model, optimizer

# 원격저장소로 체크포인트 내용 전달
def save_train_report(matrixs, model, optimizer):
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
                {"epoch": matrixs.get("epoch")},
                os.path.join(temp_checkpoint_dir, "extra_state.pt"),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        train.report(matrixs, checkpoint=checkpoint)


def run_train(s3_path, train_func, epoch, workers, run_config, valid):
    if s3_path == None:
        trainer = ray_torch.TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={"num_epochs": epoch},
            scaling_config=ScalingConfig(num_workers=workers, use_gpu=False),
            run_config=run_config,
        )
        result = trainer.fit()
        # s3_path = result.checkpoint # 사실상 db에 result.checkpoint 저장이 필요함.
        
    else:
        restored_trainer = ray_torch.TorchTrainer.restore(
            s3_path.path,
        )
        result = restored_trainer.fit()

    best_result = [None, 9e9]
    for candidate in result.best_checkpoints:
        if best_result[1] > candidate[1].get(valid):
            best_result[1] = candidate[1].get(valid)
            best_result[0] = f"s3://{candidate[0].path}"

    return result, best_result

if __name__ == "__main__":
    config = create_runconfig("user", "model")
    print(f"{config.storage_path}/{config.name}")
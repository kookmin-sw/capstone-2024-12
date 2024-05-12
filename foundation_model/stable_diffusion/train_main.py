from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer

import train as t
import dataset as d


def set_args(model_path, model_tmp_path, data_path, data_tmp_path, data_class):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={model_tmp_path}",
        f"--instance_images_dir={data_path}",
        f"--instance_prompt=photo of the {data_class}",
        f"--class_images_dir={data_tmp_path}",
        f"--class_prompt=photo of a {data_class}",
        "--train_batch_size=2",
        "--lr=5e-6",
        "--num_epochs=4",
        "--max_train_steps=200",
        "--num_workers=4",
    ]
    return cmd_args

def tune_model(cmd_args, user_id, model_id):
    args = t.train_arguments().parse_args(cmd_args)
    
    # Build training dataset.
    train_dataset = d.get_train_dataset(args)

    print(f"Loaded training dataset (size: {train_dataset.count()})")
    
    # Train with Ray Train TorchTrainer.
    trainer = TorchTrainer(
        t.train_fn,
        train_loop_config=vars(args),
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=args.num_workers,
            resources_per_worker={"GPU":1, "CPU":8},
        ),
        datasets={
            "train": train_dataset,
        },
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

    print(result)


if __name__ == '__main__':

    # --------------------------------
    # 백엔드 API 개발 후 수정 필수
    # data_tmp_dir 의 경우 사용자가 올린 데이터가 아니라
    #   정규화 시킬 이미지 path를 의미함 (로직 회의 필요)
    # data_class 는 파운데이션 모델 설정할 때 같이 입력 필수
    # model_tmp_dir 이 실제로 완성된 모델을 의미함
    #   이와 관련하여 로직 회의 필요
    # --------------------------------

    # 훈련을 위한 변수
    model_path = "/tmp/trained_model/stable_diffusion/models--CompVis--stable-diffusion-v1-4/snapshots/b95be7d6f134c3a9e62ee616f310733567f069ce"
    class_data_path = "/tmp/data/stable_diffusion/class_data"
    data_class = "dog"

    trained_model_path = "/tmp/trained_model/stable_diffusion"
    user_data_path = "/tmp/data/stable_diffusion/user_data"

    # 체크포인트를 위한 변수
    user_id = "admin"
    model_id = "stable-diffusion"

    args = set_args(model_path, trained_model_path, user_data_path, class_data_path, data_class)
    tune_model(args, user_id, model_id)

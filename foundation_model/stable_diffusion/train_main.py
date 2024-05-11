import generate as g
import train as t

def create_regularization_images(model_path, data_tmp_path, data_class):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={data_tmp_path}",
        f"--prompts=photo of the {data_class}",
        "--num_samples_per_prompt=200",
        "--use_ray_data",
    ]
    args = g.run_model_flags().parse_args(cmd_args)
    #g.run(args)


def tuning_model(model_path, model_tmp_path, data_path, data_tmp_path, data_class):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={model_tmp_path}",
        f"--instance_images_dir={data_path}",
        f"--instance_prompt=photo of the different {data_class}",
        f"--class_images_dir={data_tmp_path}",
        f"--class_prompt=photo of a {data_class}",
        "--train_batch_size=2",
        "--lr=5e-6",
        "--num_epochs=4",
        "--max_train_steps=200",
        "--num_workers=4",
    ]
    args = t.train_arguments().parse_args(cmd_args)
    return args


if __name__ == '__main__':

    # --------------------------------
    # 백엔드 API 개발 후 수정 필수
    # data_tmp_dir 의 경우 사용자가 올린 데이터가 아니라
    #   정규화 시킬 이미지 path를 의미함 (로직 회의 필요)
    # data_class 는 파운데이션 모델 설정할 때 같이 입력 필수
    # --------------------------------
    model_path = "/home/ubuntu/model"
    data_tmp_path = "/home/ubuntu/output"
    data_class = "dog"

    create_regularization_images(model_path, data_tmp_path, data_class)

    # --------------------------------
    # 백엔드 API 개발 후 수정 필수
    # model_tmp_dir 이 실제로 완성된 모델을 의미함
    #   이와 관련하여 로직 회의 필요
    # --------------------------------
    model_tmp_path = "/home/ubuntu/model_tmp"
    data_path = "/home/ubuntu/data"

    tuning_model(model_path, model_tmp_path, data_path, data_tmp_path, data_class)

import generate as g

def create_regularization_images(model_path, output_path, prompts):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={output_path}",
        f"--prompts={prompts}",
        "--num_samples_per_prompt=5",
    ]
    args = g.run_model_flags().parse_args(cmd_args)
    #g.run(args)


if __name__ == '__main__':

    # --------------------------------
    # 백엔드 API 개발 후 수정 필수
    # data_tmp_dir 의 경우 사용자가 올린 데이터가 아니라
    #   정규화 시킬 이미지 path를 의미함 (로직 회의 필요)
    # data_class 는 파운데이션 모델 설정할 때 같이 입력 필수
    # --------------------------------
    model_path = "/home/ubuntu/model"
    output_path = "/home/ubuntu/output"
    prompts = "A photo of a dog sitting on a bench."

    create_regularization_images(model_path, output_path, prompts)

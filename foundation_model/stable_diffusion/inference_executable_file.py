import generate

def create_regularization_images(model_path, output_path, prompts):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={output_path}",
        f"--prompts={prompts}",
        "--num_samples_per_prompt=1",
    ]
    args = generate.run_model_flags().parse_args(cmd_args)
    generate.run(args)


if __name__ == '__main__':

    model_path = "/tmp/trained_model/stable_diffusion"
    output_path = "/tmp/data/stable_diffusion/generate_data"
    prompts = "A photo of a dog sitting on a bench."

    create_regularization_images(model_path, output_path, prompts)

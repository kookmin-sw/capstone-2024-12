import hashlib
from os import path

import time
import torch
import ray

from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
import torch

import argparse


def run_model_flags():
    """Commandline arguments for running a tuned DreamBooth model."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Directory of the tuned model files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        required=True,
        help="Comma separated prompt strings for generating the images.",
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each prompt.",
    )
    parser.add_argument(
        "--use_ray_data",
        default=False,
        action="store_true",
        help=(
            "Enable using Ray Data to use multiple GPU workers to perform inference."
        ),
    )
    parser.add_argument(
        "--lora_weights_dir",
        default=None,
        help=("The directory where `pytorch_lora_weights.bin` is stored."),
    )

    return parser


def load_lora_weights(unet, text_encoder, input_dir):
    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
    LoraLoaderMixin.load_lora_into_unet(
        lora_state_dict, network_alphas=network_alphas, unet=unet
    )
    LoraLoaderMixin.load_lora_into_text_encoder(
        lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder
    )
    return unet, text_encoder


def get_pipeline(model_dir, lora_weights_dir=None):
    pipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    if lora_weights_dir:
        unet = pipeline.unet
        text_encoder = pipeline.text_encoder
        print(f"Loading LoRA weights from {lora_weights_dir}")
        unet, text_encoder = load_lora_weights(unet, text_encoder, lora_weights_dir)
        pipeline.unet = unet
        pipeline.text_encoder = text_encoder
    return pipeline


def run(args):
    class StableDiffusionCallable:
        def __init__(self, model_dir, output_dir, lora_weights_dir=None):
            print(f"Loading model from {model_dir}")
            self.pipeline = get_pipeline(model_dir, lora_weights_dir)
            self.pipeline.set_progress_bar_config(disable=True)
            if torch.cuda.is_available():
                self.pipeline.to("cuda")
            self.output_dir = output_dir

        def __call__(self, batch):
            filenames = []
            for i, prompt in zip(batch["idx"], batch["prompt"]):
                # Generate 1 image at a time to reduce memory consumption.
                for image in self.pipeline(prompt).images:
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = path.join(self.output_dir, f"{i}-{hash_image}.jpg")
                    image.save(image_filename)
                    print(f"Saved {image_filename}")
                    filenames.append(image_filename)
            return {"filename": filenames}

    prompts = args.prompts.split(",")

    start_time = time.time()
    num_samples = len(prompts) * args.num_samples_per_prompt

    if args.use_ray_data:
        # Use Ray Data to perform batch inference to generate many images in parallel
        prompts_with_idxs = []
        for prompt in prompts:
            prompts_with_idxs.extend(
                [
                    {"idx": i, "prompt": prompt}
                    for i in range(args.num_samples_per_prompt)
                ]
            )

        prompt_ds = ray.data.from_items(prompts_with_idxs)
        num_workers = 4

        # Run the batch inference by consuming output with `take_all`.
        prompt_ds.map_batches(
            StableDiffusionCallable,
            compute=ray.data.ActorPoolStrategy(size=num_workers),
            fn_constructor_args=(args.model_dir, args.output_dir),
            num_gpus=1,
            batch_size=num_samples // num_workers,
        ).take_all()

    else:
        # Generate images one by one
        stable_diffusion_predictor = StableDiffusionCallable(
            args.model_dir, args.output_dir, args.lora_weights_dir
        )
        for prompt in prompts:
            for i in range(args.num_samples_per_prompt):
                stable_diffusion_predictor({"idx": [i], "prompt": [prompt]})

    elapsed = time.time() - start_time
    print(
        f"Generated and saved {num_samples} images to {args.output_dir} in "
        f"{elapsed} seconds."
    )


if __name__ == "__main__":
    args = run_model_flags().parse_args()
    run(args)

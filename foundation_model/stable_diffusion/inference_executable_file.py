import hashlib
from os import path

import torch

from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin


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

class StableDiffusionCallable:
    def __init__(self, model_dir, lora_weights_dir=None):
        print(f"Loading model from {model_dir}")
        self.pipeline = get_pipeline(model_dir, lora_weights_dir)
        self.pipeline.set_progress_bar_config(disable=True)
        if torch.cuda.is_available():
            self.pipeline.to("cuda")
        self.output_dir = "/home/ubuntu/data/generate_data"

    def __call__(self, batch):
        filenames = []
        prompt = batch["prompt"]
        # Generate 1 image at a time to reduce memory consumption.
        for image in self.pipeline(prompt).images:
            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = path.join(self.output_dir, f"{hash_image}.jpg")
            image.save(image_filename)
            print(f"Saved {image_filename}")
            filenames.append(image_filename)
        return {"filename": filenames}
    

model_path = "/home/ubuntu/models"
stable_diffusion_predictor = StableDiffusionCallable(model_path)


def run(model_path, prompt):
    # Generate images one by one
    stable_diffusion_predictor({"prompt": [prompt]})


if __name__ == '__main__':
    prompt = "A photo of a dog sitting on a bench."

    result = run(model_path, prompt)
    print(type(result))
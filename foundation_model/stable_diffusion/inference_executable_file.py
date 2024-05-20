import base64
from io import BytesIO

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

    def __call__(self, prompt):
        # Generate 1 image at a time to reduce memory consumption.
        for image in self.pipeline(prompt).images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
# (수정하는 부분)
model_path = "/home/ubuntu/models"
stable_diffusion_predictor = StableDiffusionCallable(model_path)


if __name__ == '__main__':
    # (수정하는 부분)
    prompt = "A photo of a dog sitting on a bench."
    result = stable_diffusion_predictor(prompt)
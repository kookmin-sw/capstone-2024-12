from typing import Dict

import itertools
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)

# LoRA related imports begin ##
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

# LoRA related imports end ##
from diffusers.utils.import_utils import is_xformers_available
from ray.train import ScalingConfig, Checkpoint, RunConfig, FailureConfig, CheckpointConfig
from ray import train
from ray.train.torch import TorchTrainer

from ray.data import read_images
from torchvision import transforms

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import CLIPTextModel, AutoTokenizer

import tempfile
from os import path, makedirs
import argparse


LORA_RANK = 4


def train_arguments():
    """Commandline arguments for running DreamBooth training script."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained huggingface Stable Diffusion model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Directory where trained models or LoRA weights are saved.",
    )
    parser.add_argument(
        "--use_lora", default=False, action="store_true", help="Use LoRA."
    )
    parser.add_argument(
        "--instance_images_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory where a few images of the instance to be fine tuned "
            "into the model are saved."
        ),
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help=("Prompt for creating the instance images."),
    )
    parser.add_argument(
        "--class_images_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory where images of similar objects for preserving "
            "model priors are saved."
        ),
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        required=True,
        help=("Prompt for creating the class images."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Train batch size."
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Train learning rate.")
    parser.add_argument(
        "--num_epochs", type=int, default=4, help="Number of epochs to train."
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight for prior preservation loss.",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm."
    )
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers.")

    return parser



def get_train_dataset(args, image_resolution=512):
    """Build a Dataset for fine-tuning DreamBooth model."""
    # Load a directory of images as a Ray Dataset
    instance_dataset = read_images(args.instance_images_dir)
    class_dataset = read_images(args.class_images_dir)

    # We now duplicate the instance images multiple times to make the
    # two sets contain exactly the same number of images.
    # This is so we can zip them up during training to compute the
    # prior preserving loss in one pass.
    #
    # Example: If we have 200 class images (for regularization) and 4 instance
    # images of our subject, then we'll duplicate the instance images 50 times
    # so that our dataset looks like:
    #
    #     instance_image_0, class_image_0
    #     instance_image_1, class_image_1
    #     instance_image_2, class_image_2
    #     instance_image_3, class_image_3
    #     instance_image_0, class_image_4
    #     instance_image_1, class_image_5
    #     ...
    dup_times = class_dataset.count() // instance_dataset.count()
    instance_dataset = instance_dataset.map_batches(
        lambda df: pd.concat([df] * dup_times), batch_format="pandas"
    )

    # Load tokenizer for tokenizing the image prompts.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
        subfolder="tokenizer",
    )

    def _tokenize(prompt):
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()

    # Get the token ids for both prompts.
    class_prompt_ids = _tokenize(args.class_prompt)[0]
    instance_prompt_ids = _tokenize(args.instance_prompt)[0]

    # START: image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomCrop(image_resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_image(
        batch: Dict[str, np.ndarray], output_column_name: str
    ) -> Dict[str, np.ndarray]:
        transformed_tensors = [transform(image).numpy() for image in batch["image"]]
        batch[output_column_name] = transformed_tensors
        return batch

    # END: image preprocessing

    # START: Apply preprocessing steps as Ray Dataset operations
    # For each dataset:
    # - perform image preprocessing
    # - drop the original image column
    # - add a new column with the tokenized prompts
    instance_dataset = (
        instance_dataset.map_batches(
            transform_image, fn_kwargs={"output_column_name": "instance_image"}
        )
        .drop_columns(["image"])
        .add_column("instance_prompt_ids", lambda df: [instance_prompt_ids] * len(df))
    )
    # END: Apply preprocessing steps as Ray Dataset operations

    class_dataset = (
        class_dataset.map_batches(
            transform_image, fn_kwargs={"output_column_name": "class_image"}
        )
        .drop_columns(["image"])
        .add_column("class_prompt_ids", lambda df: [class_prompt_ids] * len(df))
    )
    # --- Ray Data

    # We may have too many duplicates of the instance images, so limit the
    # dataset size so that len(instance_dataset) == len(class_dataset)
    final_size = min(instance_dataset.count(), class_dataset.count())

    # Now, zip the images up.
    train_dataset = (
        instance_dataset.limit(final_size)
        .repartition(final_size)
        .zip(class_dataset.limit(final_size).repartition(final_size))
    )

    print("Training dataset schema after pre-processing:")
    print(train_dataset.schema())

    return train_dataset.random_shuffle()


def collate(batch, dtype):
    """Build Torch training batch.

    B = batch size
    (C, W, H) = (channels, width, height)
    L = max length in tokens of the text guidance input

    Input batch schema (see `get_train_dataset` on how this was setup):
        instance_images: (B, C, W, H)
        class_images: (B, C, W, H)
        instance_prompt_ids: (B, L)
        class_prompt_ids: (B, L)

    Output batch schema:
        images: (2 * B, C, W, H)
            All instance images in the batch come before the class images:
            [instance_images[0], ..., instance_images[B-1], class_images[0], ...]
        prompt_ids: (2 * B, L)
            Prompt IDs are ordered the same way as the images.

    During training, a batch will be chunked into 2 sub-batches for
    prior preserving loss calculation.
    """

    images = torch.cat([batch["instance_image"], batch["class_image"]], dim=0)
    images = images.to(memory_format=torch.contiguous_format).to(dtype)

    batch_size = len(batch["instance_prompt_ids"])

    prompt_ids = torch.cat(
        [batch["instance_prompt_ids"], batch["class_prompt_ids"]], dim=0
    ).reshape(batch_size * 2, -1)

    return {
        "images": images,
        "prompt_ids": prompt_ids,  # token ids should stay int.
    }


def prior_preserving_loss(model_pred, target, weight):
    # Chunk the noise and model_pred into two parts and compute
    # the loss on each part separately.
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    # Compute instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    # Compute prior loss
    prior_loss = F.mse_loss(
        model_pred_prior.float(), target_prior.float(), reduction="mean"
    )

    # Add the prior loss to the instance loss.
    return loss + weight * prior_loss


def get_target(scheduler, noise, latents, timesteps):
    """Get the target for loss depending on the prediction type."""
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {pred_type}")


def add_lora_layers(unet, text_encoder):
    """Add LoRA layers for unet and text encoder.

    `unet` and `text_encoder` will be modified in place.

    Returns:
        The LoRA parameters for unet and text encoder correspondingly.
    """
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(
            attn_processor,
            (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0),
        ):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0
                if hasattr(F, "scaled_dot_product_attention")
                else LoRAAttnProcessor
            )

        module = lora_attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=LORA_RANK,
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    text_lora_parameters = LoraLoaderMixin._modify_text_encoder(
        text_encoder, dtype=torch.float32, rank=LORA_RANK
    )

    return unet_lora_parameters, text_lora_parameters


def load_models(config):
    """Load pre-trained Stable Diffusion models."""
    # Load all models in bfloat16 to save GRAM.
    # For models that are only used for inferencing,
    # full precision is also not required.
    dtype = torch.bfloat16

    text_encoder = CLIPTextModel.from_pretrained(
        config["model_dir"],
        subfolder="text_encoder",
        torch_dtype=dtype,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model_dir"],
        subfolder="scheduler",
        torch_dtype=dtype,
    )

    # VAE is only used for inference, keeping weights in full precision is not required.
    vae = AutoencoderKL.from_pretrained(
        config["model_dir"],
        subfolder="vae",
        torch_dtype=dtype,
    )
    # We are not training VAE part of the model.
    vae.requires_grad_(False)

    # Convert unet to bf16 to save GRAM.
    unet = UNet2DConditionModel.from_pretrained(
        config["model_dir"],
        subfolder="unet",
        torch_dtype=dtype,
    )

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    if not config["use_lora"]:
        unet_trainable_parameters = unet.parameters()
        text_trainable_parameters = text_encoder.parameters()
    else:
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        unet_trainable_parameters, text_trainable_parameters = add_lora_layers(
            unet, text_encoder
        )

    text_encoder.train()
    unet.train()

    torch.cuda.empty_cache()

    return (
        text_encoder,
        noise_scheduler,
        vae,
        unet,
        unet_trainable_parameters,
        text_trainable_parameters,
    )


def train_fn(config):

    # Load pre-trained models.
    (
        text_encoder,
        noise_scheduler,
        vae,
        unet,
        unet_trainable_parameters,
        text_trainable_parameters,
    ) = load_models(config)

    text_encoder = train.torch.prepare_model(text_encoder)
    unet = train.torch.prepare_model(unet)
    # manually move to device as `prepare_model` can't be used on
    # non-training models.
    vae = vae.to(train.torch.get_device())

    # Use the regular AdamW optimizer to work with bfloat16 weights.
    optimizer = torch.optim.AdamW(
        itertools.chain(unet_trainable_parameters, text_trainable_parameters),
        lr=config["lr"],
    )

    train_dataset = train.get_dataset_shard("train")

    # Train!
    num_train_epochs = config["num_epochs"]

    print(f"Running {num_train_epochs} epochs.")

    global_step = 0
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            load_config = {"model_dir":checkpoint_dir}
            (
                text_encoder,
                noise_scheduler,
                vae,
                unet,
                unet_trainable_parameters,
                text_trainable_parameters,
            ) = load_models(load_config)
            optimizer.load_state_dict(
                torch.load(path.join(checkpoint_dir, "optimizer.pt"))
            )
            start_epoch = (
                torch.load(path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1
            )
            global_step = (
                torch.load(path.join(checkpoint_dir, "extra_state.pt"))["step"]
            )

    results = {}
    for epoch in range(start_epoch, num_train_epochs):
        for epoch, batch in enumerate(
            train_dataset.iter_torch_batches(
                batch_size=config["train_batch_size"],
                device=train.torch.get_device(),
            )
        ):
            batch = collate(batch, torch.bfloat16)

            optimizer.zero_grad()

            # Convert images to latent space
            latents = vae.encode(batch["images"]).latent_dist.sample() * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

            # Predict the noise residual.
            model_pred = unet(
                noisy_latents.to(train.torch.get_device()),
                timesteps.to(train.torch.get_device()),
                encoder_hidden_states.to(train.torch.get_device()),
            ).sample
            target = get_target(noise_scheduler, noise, latents, timesteps)

            loss = prior_preserving_loss(
                model_pred, target, config["prior_loss_weight"]
            )
            loss.backward()

            # Gradient clipping before optimizer stepping.
            clip_grad_norm_(
                itertools.chain(unet_trainable_parameters, text_trainable_parameters),
                config["max_grad_norm"],
            )

            optimizer.step()  # Step all optimizers.

            global_step += 1
            results = {
                "epoch": epoch,
                "step": global_step,
                "loss": loss.detach().item(),
            }
            train.report(results)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if not path.exists(temp_checkpoint_dir):
                makedirs(temp_checkpoint_dir)
            torch.save(
                optimizer.state_dict(),
                path.join(temp_checkpoint_dir, "optimizer.pt"),
            )
            torch.save(
                {"epoch":epoch, "step":global_step},
                path.join(temp_checkpoint_dir, "extra_state.pt"),
            )
            if not config["use_lora"]:
                pipeline = DiffusionPipeline.from_pretrained(
                    config["model_dir"],
                    text_encoder=text_encoder,
                    unet=unet, 
                )
                pipeline.save_pretrained(temp_checkpoint_dir)
            else:
                save_lora_weights(unet, text_encoder, temp_checkpoint_dir)
            
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(results, checkpoint=checkpoint)
    # END: Training loop

    # Create pipeline using the trained modules and save it.
    if train.get_context().get_world_rank() == 0:
        if not config["use_lora"]:
            pipeline = DiffusionPipeline.from_pretrained(
                config["model_dir"],
                text_encoder=text_encoder,
                unet=unet,
            )
            pipeline.save_pretrained(config["output_dir"])
        else:
            save_lora_weights(unet, text_encoder, config["output_dir"])


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            param_name = f"{attn_processor_key}.{parameter_key}"
            attn_processors_state_dict[param_name] = parameter
    return attn_processors_state_dict


def save_lora_weights(unet, text_encoder, output_dir):
    unet_lora_layers_to_save = None
    text_encoder_lora_layers_to_save = None

    unet_lora_layers_to_save = unet_attn_processors_state_dict(unet)
    text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder)

    LoraLoaderMixin.save_lora_weights(
        output_dir,
        unet_lora_layers=unet_lora_layers_to_save,
        text_encoder_lora_layers=text_encoder_lora_layers_to_save,
    )


def set_args(model_path, trained_model_path, user_data_path, class_data_path, data_class, epoch):
    cmd_args = [
        f"--model_dir={model_path}",
        f"--output_dir={trained_model_path}",
        f"--instance_images_dir={user_data_path}",
        f"--instance_prompt=photo of the {data_class}",
        f"--class_images_dir={class_data_path}",
        f"--class_prompt=photo of a {data_class}",
        "--train_batch_size=4",
        "--lr=5e-6",
        f"--num_epochs={epoch}",
        "--num_workers=4",
    ]
    return cmd_args

def tune_model(cmd_args, user_id, model_id):
    args = train_arguments().parse_args(cmd_args)
    
    # Build training datasetrain.
    train_dataset = get_train_dataset(args)

    print(f"Loaded training dataset (size: {train_dataset.count()})")
    
    # Train with Ray Train TorchTrainer.
    trainer = TorchTrainer(
        train_fn,
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
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
            ),
            failure_config=FailureConfig(max_failures=-1) # 계속 실행하게 함
        ),
    )
    result = trainer.fit()

    print(result)


if __name__ == "__main__":
    # 훈련을 위한 변수
    model_path = "/tmp/model/stable_diffusion/models--CompVis--stable-diffusion-v1-4/snapshots/b95be7d6f134c3a9e62ee616f310733567f069ce"
    class_data_path = "/tmp/data/stable_diffusion/class_data"
    data_class = "rabbit"

    trained_model_path = "/tmp/trained_model/stable_diffusion"
    user_data_path = "/tmp/data/stable_diffusion/user_data"

    # 체크포인트를 위한 변수
    user_id = "admin"
    model_id = "stable-diffusion"
    user_epoch = 4

    args = set_args(model_path, trained_model_path, user_data_path, class_data_path, data_class, user_epoch)
    tune_model(args, user_id, model_id)

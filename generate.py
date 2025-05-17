import torch
from utils import DEVICE, OBJ_SIZE, text_encoder, voxel_to_obj
from diffusers import UNet3DConditionModel, DDPMScheduler
import numpy as np


def generate_from_text(text_prompt, model, num_inference_steps=500, voxel_size=OBJ_SIZE):
    model.eval()

    # Initialize scheduler inside the function with proper inference steps
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)

    with torch.no_grad():
        text_emb = text_encoder([text_prompt])  # shape [1, seq_len, D]

        # Start from random noise
        voxel = torch.randn(1, 1, voxel_size, voxel_size,
                            voxel_size).to(DEVICE)

        for t in scheduler.timesteps:
            pred = model(sample=voxel, timestep=t,
                         encoder_hidden_states=text_emb).sample

            voxel = scheduler.step(pred, t, voxel).prev_sample

        return voxel


if __name__ == "__main__":
    model_time = "2025_05_02_13-50-39"
    print(f"Loading model from {model_time}")
    model = UNet3DConditionModel.from_pretrained(f"models/{model_time}/diffusion_model").to(DEVICE)
    text_prompt = "sphere"
    print(f"Generating voxel for prompt: {text_prompt}")
    voxel = generate_from_text(text_prompt, model)
    voxel = voxel.squeeze().cpu().numpy()
    voxel = (voxel > 0.5).astype(np.uint8)
    voxel_to_obj(voxel, f"models/{model_time}/{text_prompt}_output.obj")
    print("Done")

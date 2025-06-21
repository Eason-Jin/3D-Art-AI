import os
import ast
import torch
from torch.utils.data import DataLoader
from diffusers import UNet3DConditionModel
from CustomDataset import CustomDataset
from utils import MAX_TIME, OBJ_SIZE, corrupt, obj_to_voxel, text_encoder, voxel_to_obj
import pandas as pd
import shutil
from tqdm import tqdm
from accelerate import Accelerator

# `accelerate launch train.py`

OUTPUT_MODEL = False
DATA_FILE = 'obj_data.csv'

def process_time_step(i, voxel_grid, filename):
    print(f'Generating {filename} at time {i}')
    result = corrupt(voxel_grid, torch.tensor(i / MAX_TIME))
    if i % 10 == 0 and OUTPUT_MODEL:
        voxel_to_obj(result.numpy(), f'generated/{filename}_{i}.obj')
    return i, result

def main():
    accelerator = Accelerator()
    device = accelerator.device

    if OUTPUT_MODEL and accelerator.is_main_process:
        if os.path.exists('generated'):
            shutil.rmtree('generated')
        os.makedirs('generated')

    if os.path.isfile(DATA_FILE):
        if accelerator.is_main_process:
            print('Loading DataFrame')
        df = pd.read_csv(DATA_FILE)

        def parse(col):
            return df[col].apply(lambda x: torch.tensor(ast.literal_eval(x)))

        df['original_data'] = parse('original_data')
        df['noisy_data'] = parse('noisy_data')

    else:
        if accelerator.is_main_process:
            columns = ['filename', 'original_data', 'noise_level', 'noisy_data']
            rows = []

            for file in os.listdir('obj'):
                if file.lower().endswith('.obj'):
                    print(f'Loading {file}')
                    voxel_grid = torch.from_numpy(obj_to_voxel(f'obj/{file}'))
                    filename = file[:-4]

                    for i in range(MAX_TIME):
                        step, result = process_time_step(i, voxel_grid, filename)
                        new_row = {
                            'filename': filename,
                            'original_data': voxel_grid.tolist(),
                            'noise_level': step,
                            'noisy_data': result.tolist(),
                        }
                        rows.append(new_row)

            print('Creating DataFrame')
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(DATA_FILE, index=False)

        accelerator.wait_for_everyone()
        df = pd.read_csv(DATA_FILE)
        df['original_data'] = df['original_data'].apply(lambda x: torch.tensor(ast.literal_eval(x)))
        df['noisy_data'] = df['noisy_data'].apply(lambda x: torch.tensor(ast.literal_eval(x)))

    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet3DConditionModel(
        sample_size=OBJ_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock3D",) * 4,
        up_block_types=("UpBlock3D",) * 4,
        cross_attention_dim=512,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    model.train()

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process):
            noisy_voxel = batch['noisy_voxel'].to(device)
            clean_voxel = batch['clean_voxel'].to(device)
            timestep = batch['noise_level'].to(device)
            descriptions = batch['description']
            text_emb = text_encoder(descriptions).to(device)

            pred = model(sample=noisy_voxel, timestep=timestep, encoder_hidden_states=text_emb).sample
            loss = loss_fn(pred, clean_voxel)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.6f}")

    if accelerator.is_main_process:
        date_time = pd.Timestamp.now().strftime("%Y_%m_%d_%H-%M-%S")
        folder_path = f'models/{date_time}'
        os.makedirs(folder_path, exist_ok=True)
        model.module.save_pretrained(f'{folder_path}/diffusion_model') if hasattr(model, "module") else model.save_pretrained(f'{folder_path}/diffusion_model')

if __name__ == '__main__':
    main()


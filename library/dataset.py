import aiohttp
import os
import tifffile 

from careamics_portfolio import PortfolioManager
# import numpy as np
# import yaml

# import logging as log

JUMP_URL = "https://zenodo.org/records/10912386/files/noisy.tiff?download=1"

# Load Datasets


async def load_jump_dataset(path):
    print("Loading Jump Dataset")
    async with aiohttp.ClientSession() as session:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        async with session.get(JUMP_URL) as response:
            # Write the content to the file in binary mode
            with open(path, "wb") as f:
                # Await and write the response content to the file
                f.write(await response.read())
    print("Jump Dataset Loaded at", path)


def load_bsd68_dataset(root_path):
    print("Loading BSD68 Dataset")
    # instantiate data portfolio manage
    portfolio = PortfolioManager()

    # and download the data
    files = portfolio.denoising.N2V_BSD68.download(root_path)
    print("BSD68 Dataset Loaded at", root_path)

# Split Datasets

def split_jump_dataset(path, split_ratio=0.8):
    # Load and Split the Dataset
    dataset = tifffile.imread(path)
    print(f"Dataset size is {len(dataset)}")
    split_idx = int(len(dataset) * split_ratio)
    train, val = dataset[:split_idx], dataset[split_idx:]
    print(f"Split the dataset at index {split_idx}/{len(dataset)}")

    # Save the Split Datasets
    dpath = path.replace('.tiff', f"_train.tiff")
    tifffile.imwrite(dpath, data=train)
    print(f"Train dataset saved to {dpath}")
    dpath = path.replace('.tiff', f"_val.tiff")
    tifffile.imwrite(dpath, data=val)
    print(f"Validation dataset saved to {dpath}")


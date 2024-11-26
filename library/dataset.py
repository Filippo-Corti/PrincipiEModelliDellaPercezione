# import argparse
# import sys
import aiohttp
import os

from careamics_portfolio import PortfolioManager
from pathlib import Path
# import numpy as np
# import yaml

# import tifffile as tiff
# import logging as log

# Load Datasets

JUMP_URL = "https://zenodo.org/records/10912386/files/noisy.tiff?download=1"


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

def split_dataset():
    pass


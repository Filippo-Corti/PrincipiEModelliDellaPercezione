import argparse
import os
import sys
import aiohttp
import os
import numpy as np

import yaml
import tifffile as tiff
import logging as log
log.basicConfig(level=log.INFO)

pattern_split_dataset = lambda path, ext, split: path.replace(ext, f"_{split}{ext}")
pattern_stats = lambda path, ext, split, channelwise: path.replace(ext, f"_{split}_mean_std_{'channelwise' if channelwise else 'global'}.npy")

def load_datasets_yml(dataset_yml="datasets.yml"):
    with open(dataset_yml, "r") as f:
        datasets = yaml.load(f, Loader=yaml.FullLoader)
    return datasets['DATASETS']

async def download_datasets(dataset_yml="datasets.yml"):
    '''
    Download datasets from the urls provided in the datasets.yml file

    Args:
    dataset_yml: str
        Path to the datasets.yml file.
        Should have the structure:
        DATASETS:
            - path: path/to/dataset
            - url: url/to/dataset

    '''    
    DATASETS = load_datasets_yml(dataset_yml)
    async with aiohttp.ClientSession() as session:
        for dataset in DATASETS:
            path = dataset["path"]
            url = dataset["url"]
            os.makedirs(path, exist_ok=True)
            async with session.get(url) as response:
                with open(path, "wb") as f:
                    f.write(await response.read())

def load_split_datasets(dataset_name, dataset_yml='datasets.yml', channelwise_stats=True):
    '''
    Load precomputed splits for a dataset defined in datasets.yml.
    Also loads statistics.

    Args:
    dataset_name: str
        Name of the dataset in the datasets.yml file
    dataset_yml: str
        Path to the datasets.yml file
    channelwise_stats: bool
        Whether to load channelwise statistics or global statistics

    Returns:
    ((Train_dataset, Val_dataset), (Train_mean, Train_std), (Val_mean, Val_std))
    '''

    yml = load_datasets_yml(dataset_yml)
    dataset = None
    for d in yml:
        if d["name"] == dataset_name:
            dataset = d
            break
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} not found in the datasets.yml file")
    ext = '.tiff' if '.tiff' in dataset["path"] else '.tif'
    train_path = pattern_split_dataset(dataset["path"], ext, "train")
    val_path = pattern_split_dataset(dataset["path"], ext, "val")
    train = tiff.imread(train_path)
    val = tiff.imread(val_path)
    train_stats = pattern_stats(dataset["path"], ext, "train", channelwise=channelwise_stats)
    val_stats = pattern_stats(dataset["path"], ext, "val", channelwise=channelwise_stats)
    train_mean, train_std = np.load(train_stats)
    val_mean, val_std = np.load(val_stats)
    return (train, val), (train_mean, train_std), (val_mean, val_std)


def load_tiff(path, take_N=-1):
    """
    Load a TIFF dataset from a path
    
    Args:
        - path: str
            Path to the dataset
        - take_N: int
            Number of images to take from the dataset. Default is -1, which means all images
    Returns:
        - dataset: np.array
            The dataset
    """
    dataset = tiff.imread(path)
    if take_N != -1:
        log.warning(f"Taking only {take_N} images from the dataset")
        dataset = dataset[:take_N]
    return dataset

def iter_tiff_batch(path: str, batch_size: int):
    """
        Loads a TIFF file and yelds batches of size batch_size.
        File is supposed to be in format [N, C, H, W] or [N, H, W]

        Args:
            - path: str
            - batch_size: int
    """
    data = tiff.imread(path)
    N = data.shape[0]
    for i in range(0, N, batch_size):
        yield data[i:i+batch_size]
    

def split_dataset(dataset_path, split_ratio=0.8, shuffle=True, take_N=-1, seed=None):
    """
    Splits a dataset into train and validation sets and saves them as TIFF files.
    Also calculates the dataset statistics and saves them as a numpy file.
    
    Args:
        - dataset_path: str
            Path to the tiff dataset
        - split_ratio: float [0.8]
            Ratio to split the dataset into train and validation
        - shuffle: bool [True]
            Whether to shuffle the dataset before splitting
        - take_N: int [-1]
            Number of images to take from the dataset. Default is -1, which means all images
        - seed: int [None]
            Seed for the random number generator.
        
                    
    """
    dataset = load_tiff(dataset_path, take_N=take_N)
    split_idx = int(len(dataset) * split_ratio)
    if shuffle:
        log.info("Shuffling dataset")
        if seed:
            log.info(f"Using seed {seed}")
            np.random.seed(seed)
        np.random.shuffle(dataset)
    train, val = dataset[:split_idx], dataset[split_idx:]

    # Saving the datasets
    ext = '.tiff' if '.tiff' in dataset_path else '.tif'
    if ext not in dataset_path:
        raise ValueError(f"Dataset path should have .tif or .tiff extension. Got {dataset_path}")

    dpath = dataset_path.replace(ext, f"_train{ext}")
    tiff.imwrite(dpath, data=train)
    log.info(f"Train dataset saved to {dpath}")
    dpath = dataset_path.replace(ext, f"_val{ext}")
    tiff.imwrite(dpath, data=val)
    log.info(f"Validation dataset saved to {dpath}")
    
    # Calculating and saving the dataset statistics.
    # Data shape is assumed to be (N, C, H, W)
    for dset, name in zip([train, val], ["train", "val"]):
        mean = np.mean(dset, axis=(0))
        std = np.std(dset, axis=(0))
        dpath = dataset_path.replace(ext, f"_{name}_mean_std_channelwise.npy")
        np.save(dpath, np.array([mean, std]))
        log.info(f"Dataset channelwise statistics saved to {dpath}")
        glob_mean = np.mean(dset, axis=(0, 1))
        glob_std = np.std(dset, axis=(0, 1))
        dpath = dataset_path.replace(ext, f"_{name}_mean_std_global.npy")
        np.save(dpath, np.array([glob_mean, glob_std]))
        log.info(f"Dataset global statistics saved to {dpath}")



async def main():
    # await download_datasets()
    if args.download:
        await download_datasets()
    # Load datasets
    datasets = load_datasets_yml()
    for dataset_descr in datasets:
        if args.split:
            split_dataset(dataset_descr["path"], split_ratio=args.split_ratio, shuffle=args.shuffle, take_N=args.N, seed=args.seed)    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='AI4Life Dataset Handler')
    parser.add_argument('--download', action='store_true', help='Download datasets')
    #Take only N first images from the dataset
    parser.add_argument('--N', type=int, default=-1, help='Number of images to take from the dataset. Default is -1, which means all images')
    parser.add_argument('--cache_augmentations', action='store_true', help='Cache augmentations')
    parser.add_argument('--split', action='store_true', help='Split dataset into train and validation')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Split ratio for the dataset')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset before splitting', default=True)
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator')

    args = parser.parse_args()


    import asyncio
    asyncio.run(main())



import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('hdn')
from hdn.models.lvae import LadderVAE
from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
import hdn.boilerplate.boilerplate as boilerplate
import hdn.utils as utils
from hdn import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import load_split_datasets, load_datasets_yml
import argparse
import logging as log
import os
from numpy.random import shuffle
import yaml
from hdn.boilerplate.boilerplate import HDNPatchCachedDataset
from torch.utils.data import DataLoader

get_cached_patch_path = lambda dset_basefolder, split, patch_size: os.path.join(dset_basefolder, f"hdn_patches_{split}_{patch_size}.npy")
get_cached_patch_shape_path = lambda dset_basefolder, patch_size: get_cached_patch_path(dset_basefolder, 'shapes', patch_size).replace('.npy', '.yml')
get_dataset_basefolder = lambda dataset_name, dataset_yml='dataset.yml': os.path.dirname([d["path"] for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0])

def cache_patches(
              dataset_name: str, 
              dataset_yml: str, 
              patch_size: int = 64,
              val_data_limit: int = 1000,
              ):
    """
        Store pre-computed augmentations patches into the same folder of the provided dataset for later use.
        Saved files are already normalized. Train data is already shuffled.
    """
    log.info(f"Pre-computing patches for {dataset_name} and patch size {patch_size}")

    (train, val), (train_mean, train_std), (val_mean, val_std) = load_split_datasets(dataset_name, dataset_yml=dataset_yml)

    log.info("Applying augmentations")
    train = utils.augment_data(train)
    img_width = train.shape[-1]
    img_height = train.shape[-2]
    num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
    train_images = utils.extract_patches(train, patch_size, num_patches)
    val_images = utils.extract_patches(val, patch_size, num_patches)
    val_images = val_images[:val_data_limit] # We limit validation patches to 1000 to speed up training but it is not necessary
    test_images = val_images[:100]
    print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)
    
    data_basefolder = get_dataset_basefolder(dataset_name=dataset_name, dataset_yml=dataset_yml)
    mean_path = get_cached_patch_path(data_basefolder, 'mean', patch_size)
    var_path = get_cached_patch_path(data_basefolder, 'var', patch_size)
    train_path = get_cached_patch_path(data_basefolder, 'train', patch_size)
    val_path = get_cached_patch_path(data_basefolder, 'val', patch_size)
    test_path = get_cached_patch_path(data_basefolder, 'test', patch_size)
    all_patches = np.concatenate((train_images, val_images, test_images), axis=0)
    patches_mean = all_patches.mean(axis=((0, 2, 3) if all_patches.ndim == 4 else None), keepdims=True)
    patches_var = all_patches.mean(axis=((0, 2, 3) if all_patches.ndim == 4 else None), keepdims=True)
    del all_patches

    log.info(f"Normalizing data...")
    train_images = (train_images-patches_mean)/patches_var
    val_images = (val_images-patches_mean)/patches_var
    test_images = (test_images-patches_mean)/patches_var
    

    log.info(f"Shuffling training data")
    shuffle(train_images)
    

    log.info(f"Train patches shape: {train_images.shape}")
    log.info(f"Val patches shape: {val_images.shape}")  
    log.info(f"Test patches shape: {test_images.shape}")

    np.save(mean_path, patches_mean)
    log.info(f"Patches mean saved to {mean_path}")
    np.save(var_path, patches_var)
    log.info(f"Patches variance saved to {var_path}")
    
    np.save(train_path, train_images)
    log.info(f"Train patches saved to {train_path}")
    np.save(val_path, val_images)
    log.info(f"Validation patches saved to {val_path}")
    np.save(test_path, test_images)
    log.info(f"Testing patches saved to {test_path}")


def load_cached_patches(dataset_name: str, dataset_yml: str, memmapped=False, splits: list[str]=["train", "val", "test"], patch_size: int=64):
    """
        Load pre-computed patches.
        
        Returns:
            - tuple: memmapped arrays corresponding to the ones specified in "splits"
            - channelwise mean of the dataset
            - channelwise variance of the dataset
    """
    data_basefolder = get_dataset_basefolder(dataset_name=dataset_name, dataset_yml=dataset_yml)
    paths = (get_cached_patch_path(data_basefolder, s, patch_size) for s in splits)
    
    images = (np.load(p, 
                        mmap_mode='r' if memmapped else None
                        )
                for p in paths)



    # Load Statistics
    mean_path = get_cached_patch_path(data_basefolder, 'mean', patch_size=patch_size)
    var_path = get_cached_patch_path(data_basefolder, 'var', patch_size=patch_size)

    return images, np.load(mean_path), np.load(var_path)






def train_hdn(
              output_root:str, 
              dataset_name: str, 
              dataset_yml: str, 
              noise_model: str,
              patch_size: int = 64,
              batch_size: int = 64,
              memload_dataset: bool = False,
              take_only: int = None,
              virtual_batch: int = 8,
              max_epochs: int = 500,
              lr=3e-4,
              steps_per_epoch = 400,
              test_batch_size=100,
              channel: int = 0,
              ):
    
    """
        Trains an HDN model on the provided dataset and using the given noise model.

        Args:
            - output_root:
            - dataset_name:
            - dataset_yml:
            - noise_model: 
            - patch_size: int (Default: 64)
                We extract overlapping patches of size ```patch_size x patch_size``` from training and validation images.
                Usually 64x64 patches work well for most microscopy datasets
            - val_data_limit: int (Default: 1000)
                Limit validation data samples to speedup training.
            - batch_size: int
            - memload_dataset: bool
                Whether to load the full dataset in memory. Speeds up training if enough RAM is available. 
            - take_only: int
                Only use this number of training patches. Useful for debugging.

    """
    
    log.info(f"Setting up training of HDN for {dataset_name} using the {noise_model} noise model.")
    
    model_name = f"hdn_{noise_model}_noisemodel"
    directory_path = os.path.join(output_root, dataset_name, 'hdn', f'channel_{channel}')
    log.info(f"Model name: {noise_model} \n Output path: {directory_path}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #try:
    (train_images, val_images, test_images), data_mean, data_var = load_cached_patches(dataset_name=dataset_name,
                                dataset_yml=dataset_yml,
                                memmapped=not memload_dataset,
                                splits=['train', 'val', 'test'],
                                patch_size = patch_size,
                                )
    if train_images.ndim == 3:
        print(f"Dataset is single-channel")
    elif train_images.ndim == 4:
        print(f"Dataset is multi-channel, taking channel {channel}")
        train_images = train_images[:, channel, ...]  
        val_images = val_images[:, channel, ...] 
        test_images = test_images[:, channel, ...]
        data_mean = data_mean[:, channel, ...]
        data_var = data_var[:, channel, ...]

    if take_only:
        log.warn(f"Taking only {take_only} samples for training!")
        train_images = train_images[:take_only]
        
    data_mean = torch.from_numpy(data_mean).to(device)
    data_std = np.sqrt(data_var)
    data_std = torch.from_numpy(data_std).to(device)
        
    img_shape = (train_images.shape[-2], train_images.shape[-1])
  
    # Data-specific
    gaussian_noise_std = None
    noise_model_path = f"noise_models/{dataset_name}/{noise_model}/channel_{channel}/GMM.npz"
    log.info(f"Loading noise model from {noise_model_path}")
    noise_model_params= np.load(noise_model_path)
    noiseModel = GaussianMixtureNoiseModel(params = noise_model_params, device = device)
    # Training-specific


    # Model-specific
    num_latents = 6
    z_dims = [32]*int(num_latents)
    blocks_per_layer = 5
    batchnorm = True
    free_bits = 1.0
    use_uncond_mode_at=[0,1]

    train_loader = DataLoader(HDNPatchCachedDataset(train_images), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(HDNPatchCachedDataset(val_images), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HDNPatchCachedDataset(test_images), batch_size=test_batch_size, shuffle=False)


    model = LadderVAE(z_dims=z_dims,
                    blocks_per_layer=blocks_per_layer,
                    data_mean=data_mean,
                    data_std=data_std,
                    noiseModel=noiseModel,
                    device=device,
                    batchnorm=batchnorm,
                    free_bits=free_bits,
                    img_shape=img_shape,
                    use_uncond_mode_at=use_uncond_mode_at).cuda()

    model.train() # Model set in training mode

    training.train_network(model=model,
                        lr=lr,
                        max_epochs=max_epochs,
                        steps_per_epoch=steps_per_epoch,
                            directory_path= directory_path,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            virtual_batch=virtual_batch,
                            gaussian_noise_std=gaussian_noise_std,
                            model_name=model_name, 
                            val_loss_patience=30)


if __name__ == "__main__":
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser(description="Train an HDN model from a noisy dataset and a given ground truth generated from another model, e.g., N2V/N2V2")
    parser.add_argument('--output_root', type=str, default='models/')
    parser.add_argument('--cache_patches', action='store_true', help="Cache the datasets patches in the dataset folder for later use. This is needed to be able to train.")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use as observations (the full dataset will be used).')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml"),
    parser.add_argument('--noise_model_name', type=str, help='Name of the noise model to train HDN.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch Size for both augmentation and model training.')
    parser.add_argument('--memload_dataset', action='store_true', help='Whether to load the full dataset in memory. If not specified, a memory mapped array will be used')
    parser.add_argument('--take_only', type=int, default=None, help='Only uses the given number of patches. Useful for debugging purposes.')
    parser.add_argument('--virtual_batch', type=int, default=8, help='Virtual Batch Size.')
    parser.add_argument('--max_epochs', type=int, default=500, help="Max epochs")
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--steps_per_epoch', type=int, default=400, help="Steps per Epoch")
    parser.add_argument('--test_batch_size', type=int, default=100, help='Test Batch Size')
    parser.add_argument('--channel', type=int, default=0, help="Which channel to train HDN on. Defaults to 0.")




    args = parser.parse_args()
    if args.cache_patches:
        cache_patches(dataset_name=args.dataset_name,
                      dataset_yml=args.dataset_yml,
                      patch_size=args.patch_size,
                      val_data_limit=1000)
    else:
        train_hdn(
                    output_root=args.output_root,
                    dataset_name=args.dataset_name,
                    dataset_yml=args.dataset_yml,
                    noise_model=args.noise_model_name,
                    batch_size = args.batch_size,
                    patch_size=args.patch_size,
                    memload_dataset=args.memload_dataset,
                    take_only=args.take_only,
                    virtual_batch=args.virtual_batch,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    steps_per_epoch=args.steps_per_epoch,
                    test_batch_size=args.test_batch_size,
                    channel=args.channel,
                )

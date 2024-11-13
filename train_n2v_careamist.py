import argparse
import logging as log
import careamics as cm
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from careamics import CAREamist
from careamics.utils.metrics import psnr
from careamics.config import create_n2v_configuration

from datasets import load_split_datasets
import os

# CPU COUNT
log.basicConfig(level=log.INFO)

def main(dataset_name, batch_size=256, take_n=-1, use_n2v2=False, output_root='models/', epochs=1000, independent_channels=True, noflip=False, norot=0):
    """
    Train a Noise2Void model on the given dataset.

    Args:
    dataset_name (str): Name of the dataset to use.
    batch_size (int): Batch size for training.
    take_n (int): Number of images to take from the dataset. Default is -1, which means all images.
    use_n2v2 (bool): Use Noise2Void2 model.
    output_root (str): Output ROOT directory for the model.
    epochs (int): Number of epochs to train the model.
    input_channel (int): Input channel to use. Default is None, which means all channels.
    """

    model_name = 'n2v2'if use_n2v2 else 'n2v'

    log.info(f"Training Noise2Void model on dataset {dataset_name} with batch size {batch_size} and model {model_name}.")
    log.info(f"Model will be saved to {output_root}")

    if take_n > 0:
        log.warning(f"Taking only the first {take_n} samples from the dataset.")

    (train, val), (train_mean, train_std), (val_mean, val_std) = load_split_datasets(dataset_name)
    
    if take_n > 0:
        train = train[:take_n]
        val = val[:take_n]
    
    log.info(f"Got train shape: {train.shape}")
    log.info(f"Got val shape: {val.shape}")

    exp_name = f'{model_name}_{dataset_name}{"_chwise" if independent_channels else ""}'

    config = create_n2v_configuration(
                                    experiment_name=exp_name,
                                    data_type='array',
                                    axes='SCYX' if train.ndim == 4 else 'SYX',
                                    n_channels=train.shape[1] if train.ndim == 4 else 1,
                                    patch_size=(64, 64),
                                    batch_size=batch_size,
                                    num_epochs=epochs,
                                    independent_channels=independent_channels,
                                    )

    if noflip:
        config.data_config.transforms[0].flip_y = False  # do not flip y
    if norot:
        config.data_config.transforms.pop(1)  # remove 90 degree rotations

    careamist = CAREamist(source=config, work_dir=os.path.join(output_root, exp_name))

    # train model
    log.info(f"Training starting now... {config}")
    careamist.train(train_source=train, val_source=val)
    log.info("Training ended!")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='AI4Life Dataset Handler')
    argparser.add_argument('--dataset_name', type=str, help='Name of the dataset to use.')
    argparser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    argparser.add_argument('--take_n', type=int, default=-1, help='Number of images to take from the dataset. Default is -1, which means all images.')
    argparser.add_argument('--use_n2v2', action='store_true', help='Use Noise2Void2 model.')
    argparser.add_argument('--output_dir', type=str, default='models/', help='Output ROOT directory for the model.') 
    argparser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model.')
    argparser.add_argument('--joint_channels', action='store_true', help='Use joint channels. Default is independent channels.')
    argparser.add_argument('--noflip', action='store_true', help="Do not flip the images")
    argparser.add_argument('--norot', action='store_true', help="Do not 90-rotate the images")
    
    args = argparser.parse_args()

    main(args.dataset_name, args.batch_size, args.take_n, args.use_n2v2, output_root=args.output_dir, epochs=args.epochs, independent_channels=not args.joint_channels, noflip=args.noflip, norot=args.norot)
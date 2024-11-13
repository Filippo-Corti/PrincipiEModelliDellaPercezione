from careamics import CAREamist
import careamics.dataset.tiling as tiling
from careamics.prediction_utils import stitch_prediction_single
import torch
import tifffile
import sys
sys.path.append('hdn')
from boilerplate import boilerplate

import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import argparse
import glob
from pathlib import Path
from skimage.util import view_as_windows


def save_result_image_tiff(image_array: np.ndarray, result_path: Path):
    # Taken from Challenge inference.py
    print(f"Writing an image to: {result_path}")
    with tifffile.TiffWriter(result_path) as out:
        out.write(
            image_array,
            resolutionunit=2
        )

def batch_iterator(array, batch_size):
    """
    Generator function to yield batches of the array.

    Parameters
    ----------
    array : numpy array
        The array to iterate over in batches.
    batch_size : int
        The size of each batch.

    Yields
    ------
    numpy array
        A batch of the array.
    """
    total_batches = int(np.ceil(array.shape[0] / batch_size))
    for i in range(total_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, array.shape[0])
        yield array[start_index:end_index]

def predict_hdn_patches(image: np.ndarray, model, device:str, patch_size: int=64, patch_batch_size:int =32):
    H, W = image.shape
    tiles_generator = tiling.extract_tiles(arr=image[None, None, ...], tile_size=(patch_size, patch_size), overlaps=(patch_size//2 + 1, patch_size//2 + 1))
    
    
    pred_patches = list()
    patch_info = list()
    current_batch = list()
    while True:
        try:
            for b in range(patch_batch_size):
                current_tile, current_info = next(tiles_generator)
                patch_info.append(current_info)
                current_batch.append(current_tile)
        except StopIteration:
            break
        x = torch.Tensor(np.array(current_batch))

        # Average 10 samples from the model to reduce variance between the patches
        current_patches = np.mean(np.array([boilerplate.predict_sample(x, model, None, device=device) for s in range(10)]), axis=0)

        pred_patches += list(current_patches)
        current_batch = list()
    print(len(pred_patches))
    pred_patches = list(np.array(pred_patches)[:, None, ...])
    return stitch_prediction_single(pred_patches, patch_info)

def predict_hdn(input_path:str, model_ckpt:str, batch_size: str, use_tta=False, patch_size=None, patch_batch_size=32):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    all_files = glob.glob(os.path.join(input_path, '*.tif*'))
    print(f"Found {len(all_files)} files to predict.")
    
    for tiff_in_path in all_files:
        if 'unstructured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('unstructured-noise', 'denoised')
        elif 'structured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('structured-noise', 'denoised')
        else:
            raise ValueError(f"Path {tiff_in_path} does not contain the correct patern 'image-stack-[un]structured-noise'") 
        tiff_out_path = tiff_out_path.replace("input/", "output/")
        print(f"Predicting file {tiff_in_path}...")
        tiff_input = tifffile.imread(tiff_in_path)
        print(f"File Shape: {tiff_input.shape}")

        print(f"Predicting one frame at a time to avoid OOMs...")
        
        if tiff_input.ndim == 3:
            tiff_input = tiff_input[:, None, ...]
        # Fix the number of dimensions
        # Iterate over channels
        N,C,H,W = tiff_input.shape

        models = []
        for c in range(C):
            m = torch.load(model_ckpt.replace("channel_0", f"channel_{c}"))
            m.mode_pred=True
            m.eval()
            models.append(m)
        
        tiff_output = None
        for img in tiff_input: # img has shape (C, H, W)
            out_channels = None
            for c in range(C):
                img_c = img[c, ...] # img_c has shape (H, W)
                img_c = (img_c - img_c.mean()) / img_c.std()
                if patch_size is None:
                    out_image = boilerplate.predict_sample(torch.Tensor(img_c[None, None, ...]), models[c], None, device=device)
                else:
                    out_image = predict_hdn_patches(img_c, models[c], device=device, patch_size=patch_size, patch_batch_size=patch_batch_size)
                out_channels = out_image if out_channels is None else np.concatenate([out_channels, out_image], axis=1)
            print(f"Img Shape: {out_channels.shape}")
            tiff_output = out_channels if tiff_output is None else np.concatenate([tiff_output, out_channels], axis=0)
            print(f"Current TIFF Output: {tiff_output.shape}")
        tiff_output = tiff_output.squeeze()

        print(f"Final Prediction Shape: {out_channels.shape}")
        os.makedirs(os.path.dirname(tiff_out_path), exist_ok=True)
        save_result_image_tiff(tiff_output, Path(tiff_out_path))
        print(f"Prediction written to {tiff_out_path}")
    print(f"Done")

def predict_n2v(input_path:str, model_ckpt:str, batch_size: str):
    all_files = glob.glob(os.path.join(input_path, '*.tif*'))
    print(f"Found {len(all_files)} files to predict.")
    model = CAREamist(model_ckpt, work_dir=os.path.dirname(os.path.dirname(model_ckpt)))

    for tiff_in_path in all_files:
        if 'unstructured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('unstructured-noise', 'denoised')
        elif 'structured-noise' in tiff_in_path:
            tiff_out_path = tiff_in_path.replace('structured-noise', 'denoised')
        else:
            raise ValueError(f"Path {tiff_in_path} does not contain the correct patern 'image-stack-[un]structured-noise'") 
        tiff_out_path = tiff_out_path.replace("input/", "output/")
        print(f"Predicting file {tiff_in_path}...")
        tiff_input = tifffile.imread(tiff_in_path)
        print(f"File Shape: {tiff_input.shape}")
        pred_batch = list()
        print(f"Predicting one frame at a time to avoid OOMs...")
        for img in tiff_input:
            if img.ndim == 2 or img.ndim == 3:
                img = img[None, ...]
            print(f"Img Shape: {img.shape}")
            pred_batch += model.predict(source=img, data_type='array', axes='SCYX' if img.ndim == 4 else 'SYX')
        pred_batch = np.concatenate(pred_batch, axis=0)
        print(f"Final Prediction Shape: {pred_batch.shape}")
        os.makedirs(os.path.dirname(tiff_out_path), exist_ok=True)
        save_result_image_tiff(pred_batch, Path(tiff_out_path))
        print(f"Prediction written to {tiff_out_path}")
    print(f"Done")

def predict_challenge(input_path: str, model_ckpt:str, model_name: str, batch_size: int, args):
    if model_name.lower() in ['n2v', 'n2v2', 'hdn']:
        print(f"Predicting using:")
        print(f"Input: {input_path}")
        print(f"Checkpoint: {model_ckpt}")
        print(f"Model: {model_name}")
        print(f"Batch Size: {batch_size}")
        print(f"Use TTA (HDN): {args.use_tta}")
        print(f"Patch size: {args.patch_size}")
        print(f"Patch Batch size: {args.patch_batch_size}")
        
        if model_name.lower() in ['n2v', 'n2v2']:
            predict_n2v(input_path=input_path,
                        model_ckpt=model_ckpt,
                        batch_size=batch_size)
        elif model_name.lower() in ['hdn']:
            predict_hdn(input_path=input_path,
                        model_ckpt=model_ckpt,
                        batch_size=batch_size,
                        use_tta=args.use_tta,
                        patch_size=args.patch_size,
                        patch_batch_size=args.patch_batch_size)
    else:
        raise NotImplementedError(f"Model name {model_name} not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict for AI4Life Denoising Challenge")
    parser.add_argument('--input_path', type=str, help="Input folder containing .tiff images")
    parser.add_argument('--model_ckpt', type=str, help="Checkpoint of the model to use")
    parser.add_argument('--model_name', type=str, help="Name of the model to use, it is used to choose implementation. Can be either 'n2v', 'n2v2', 'hdn'")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size to use")
    parser.add_argument('--use_tta', action='store_true', help='Use TTA for HDN')
    parser.add_argument('--patch_size', type=int, default=64, help="Do patchwise predictions for HDN. Requires less memory but introduces artefacts.")
    parser.add_argument('--patch_batch_size', type=int, default=32, help="How many patches at a time to consume when patchise inference is active for HDN.")

    args = parser.parse_args()

    predict_challenge(input_path=args.input_path,
                      model_ckpt=args.model_ckpt,
                      model_name=args.model_name,
                      batch_size=args.batch_size,
                      args=args)
    
    
    
import torch
from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from hdn.lib import histNoiseModel
from hdn.lib.utils import plotProbabilityDistribution
from datasets import load_datasets_yml
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging as log

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

log.basicConfig()
log.getLogger().setLevel(log.INFO)

def plotProbabilityDistribution(ax1, ax2, signalBinIndex, histogram, gaussianMixtureNoiseModel, min_signal, max_signal, n_bin, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal."""
    histBinSize = (max_signal - min_signal) / n_bin
    querySignal_numpy = (signalBinIndex / float(n_bin) * (max_signal - min_signal) + min_signal)
    querySignal_numpy += histBinSize / 2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)
    
    queryObservations_numpy = np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy += histBinSize / 2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch = gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy = pTorch.cpu().detach().numpy()
    
    ax1.clear()
    ax2.clear()
    
    ax1.set_xlabel('Observation Bin')
    ax1.set_ylabel('Signal Bin')
    ax1.imshow(histogram**0.25, cmap='gray')
    ax1.axhline(y=signalBinIndex + 0.5, linewidth=5, color='blue', alpha=0.5)
    
    ax2.plot(queryObservations_numpy, histogram[signalBinIndex, :] / histBinSize, label='GT Hist: bin =' + str(signalBinIndex), color='blue', linewidth=2)
    ax2.plot(queryObservations_numpy, pNumpy, label='GMM : ' + ' signal = ' + str(np.round(querySignal_numpy, 2)), color='red', linewidth=2)
    ax2.set_xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    ax2.set_ylabel('Probability Density')
    ax2.set_title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    ax2.legend()

def generate_noise_model(
                            output_root:str, 
                            gt_path:str, 
                            gt_name:str, 
                            dataset_name: str, 
                            dataset_yml: str,
                            n_gaussian: int = 3,
                            n_coeff: int = 2,
                            bins: int = 250,
                            signal_bin_index: int = 100,
                            epochs: int = 2000,
                            channel: int = None
                            ):

    device = torch.device("cuda:0")

    dset = [d for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0]

    gt_path = gt_path or f'predictions/{dataset_name}/{gt_name}.tiff'
    
    log.info(f"Loading signals...")
    full_observations = tifffile.imread(dset['path'])
    full_signal = tifffile.imread(gt_path).squeeze()
    
    
    channels = range(full_signal.shape[1]) if full_signal.ndim == 4 else [0]

    for channel in channels: 
        
        # assert (channel is None and full_signal.ndim == 3) or (channel is not None and full_signal.ndim == 4), "If image is multichannel then select a channel, otherwise set channel to None."
        if channel is not None:
            print(f"Selecting channel {channel}")
            observations = full_observations[:, channel, ...]
            signal = full_signal[:, channel, ...]
        
        out_root = os.path.join('noise_models', dataset_name, gt_name, f'channel_{channel if channel is not None else 0}')
        datamin, datamax = observations.min(), observations.max()
        log.info(f"Done.")

        try:
            log.info(f"Generating histogram...")
            histogram = histNoiseModel.createHistogram(bins=bins, minVal=datamin, maxVal=datamax, observation=observations, signal=signal)
            os.makedirs(out_root, exist_ok=True)
            np.save(os.path.join(out_root, 'histogram.npy'), histogram)
            histogramFD = histogram[0]

            # Let's look at the histogram-based noise model.
            plt.xlabel('Observation Bin')
            plt.ylabel('Signal Bin')
            plt.imshow(histogramFD**0.25, cmap='gray')
            plt.show()
            min_signal=np.percentile(signal, 0.5)
            max_signal=np.percentile(signal, 99.5)
            log.info(f"Minimum Signal Intensity is {min_signal}")
            log.info(f"Maximum Signal Intensity is {max_signal}")

            log.info(f"Training GMM for {dataset_name} with {gt_path} GT:")
            gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal, max_signal = max_signal, path=out_root+'/', weight = None, n_gaussian = n_gaussian, n_coeff = n_coeff, device = device, min_sigma = 50)
            gaussianMixtureNoiseModel.train(signal, observations, batchSize = 250000, n_epochs = epochs, learning_rate = 0.1, name = 'GMM', lowerClip = 0.5, upperClip = 99.5)
            log.info(f"Model saved Successfully")
            print(f"Created Noise Model for {gt_name} {dataset_name} at {out_root}")
        
        except Exception as e:
            log.error(f"ERROR PROCESSING DATASET {dataset_name} and model {gt_name}: {str(e)}")


        try:
            # Initialize figure and axes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Animation function
            def animate(i):
                plotProbabilityDistribution(ax1, ax2, signalBinIndex=i, histogram=histogramFD, gaussianMixtureNoiseModel=gaussianMixtureNoiseModel, min_signal=datamin, max_signal=datamax, n_bin=bins, device=device)
                return ax1, ax2

            # Create the animation
            ani = animation.FuncAnimation(fig, animate, frames=bins, interval=200, blit=False)

            # Save the animation as a GIF
            ani.save(os.path.join(out_root, 'animation.gif'), writer='pillow')
            
        except Exception as e:
            log.error(f"Failed to write animation gif: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Noise Model from a noisy dataset and a given ground truth generated from another model, e.g., N2V/N2V2")
    parser.add_argument('--output_root', type=str, default="noise_models/", help='Output root folder to store noise models. Each Noise model will be saved in <output_root>/<dataset_name>/<gt_name>/[GMM.npz|histogram.npy]')
    parser.add_argument('--gt_name', type=str, help='Name of the GT used to learn the model. If bootstrapping, this could be the name of the model used to generate the GTs. This is used to build the save path for noise models.')
    parser.add_argument('--gt_path', type=str, help='Path of the TIFF ground truth (or upstream model predictions, if bootstrapping) to use for generating the noise model. If None, the default is predictions/<dataset_name>/<gt_name>.tiff')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to use as observations (the full dataset will be used).')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml")
    parser.add_argument('--n_gaussian', type=int, default=3, help='Number of gaussians to use for Gaussian Mixture Model')
    parser.add_argument('--n_coeff', type=int, default=2, help='No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.')
    parser.add_argument('--bins', type=int, default=250, help='Number of bins to use for geneating the histogram.')
    parser.add_argument('--signal_bin_index', type=int, default=100, help='Bin index of the signal that is shown in the saved PNG image')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of Epochs to train GMM.')   

    args = parser.parse_args()
    generate_noise_model(output_root=args.output_root,
                         gt_path=args.gt_path,
                         gt_name=args.gt_name,
                         dataset_name=args.dataset_name,
                         dataset_yml=args.dataset_yml,
                         epochs=args.epochs
                        )
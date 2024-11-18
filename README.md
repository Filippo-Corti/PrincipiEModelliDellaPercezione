# Step-by-step guide to Setup and Run N2V Predictions

## 1. Check/install dependencies 

- Make sure Python and Anaconda ("conda") are installed and working.
- Setup the Conda Environment:

    ``` conda env create -f conda.yml ```

    ``` conda activate n2v ```

    ``` pip install -r requirements.txt ```

> This apparently simple step took me 3.5 hours of conversations with ChatGPT and only worked on Linux (not on Windows)

## 2. Download JUMP Dataset

- First Download the Dataset from the URL using:
    ``` python datasets.py --download ```

- Then Split the Dataset between Training and Validation Data using:
``` python datasets.py --split --split_ratio=0.8 --shuffle --seed=0 ```

> This will also create some .npy files containing some statistics about the data (unsure about the usefulness of this)

### Important: 
The original noisy.tiff file is composed of 517x4=2068 layers, but there are only 517 images, because every image is stored as 4 grayscale images representing (probably) the RGBA channels. This type of tiff file is called hyperstack. \
\
The split operation merges the 4 channels for every image (that's the way tifffile works apparently), considering the 517 colored images. \
Because the images are only 517, they are distributed with a split_ratio of 0.8 as follows:
* 413 images in noisy_train.tiff
* 104 images in noisy_val.tiff

## 3. Training the N2V Model

- For some reason a new package is needed. Run:

    ``` pip install lightning[extra] ```

- Train the N2V Model using:

``` python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name jump_cell_painting ```

> Currently run with epochs = 100 and batch_size = 32

## 4. Apply the Model to the Dataset

- Run the command:

``` python generate_n2v_predictions.py --model_name=n2v --model_ckpt=models/n2v_n2v2/n2v_jump_cell_painting_chwise/checkpoints/last.ckpt --dataset_name=jump_cell_painting ```

> For some reason it prints "Killed" after there are no more images to load. 
> It's probably an issue with the storage of big files. Currently it stops at 100 images.

## 5. Split Predicted Images into Channels

- Predictions produced by the Algorithm show the 4 channels merged in a single picture. Split the channels with:

``` python split_predictions_channels.py ```

> Splitting the Predictions is not needed but allows for a better comparison with the original Dataset.
# Neural Network Denoising with the Noise2Void Algorithm

![Grand Challenge Banner](https://rumc-gcorg-p-public.s3.amazonaws.com/b/756/denoising.x10.jpeg)

This repository provides a step-by-step guide for training and testing a U-Net Neural Network for image denoising using:
* The [JUMP Dataset](https://github.com/jump-cellpainting/datasets). 
* The [BSD68 Benchmark Dataset](https://paperswithcode.com/dataset/bsd)

The network implements the Noise2Void (N2V) algorithm, as available through the [CAREamics](https://careamics.github.io/0.1/) library. This approach allows for denoising without requiring ground truth clean images, making it particularly suited for applications where clean references are unavailable.


## Project Goals
* Explore the potential of Deep Neural Networks (DNNs) for image denoising.
* Compare results of N2V with those obtained using traditional denoising methods.
* Understand the inner workings of the N2V algorithm and its implementation details.


## Steps to Reproduce

### 1. Setup a Virtual Environment using Anaconda 

- Make sure Python and Anaconda ("conda") are installed and working.
- Setup the Conda Environment:

    ``` conda env create -f conda.yml ```

    ``` conda activate n2v ```

    ``` pip install -r requirements.txt ```

### 2. Run the scripts

- Follow the Jupyter Notebooks to Train the Model and Generate some Predictions:  

    ``` jump_cells.ipynb ```

    ``` bsd68.ipynb ```




import tifffile
import numpy as np
import os

def split_channels_to_single_stack(input_file: str, output_file: str):
    """
    Splits a TIFF hyperstack with multiple channels into a single stack of images.
    Each group of 4 consecutive images in the output corresponds to the 4 channels of one input image.
    
    Args:
        input_file (str): Path to the input TIFF file.
        output_file (str): Path to save the output TIFF file.
    """
    # Load the hyperstack
    data = tifffile.imread(input_file)
    print(f"Loaded data shape: {data.shape}")

    # Check data shape (expected: (N, C, Y, X))
    if data.ndim != 4:
        raise ValueError("Input data must be a 4D array (N, C, Y, X).")

    # Split channels and preserve image layout
    num_images, num_channels, height, width = data.shape

    # We will create an output stack where each 4 consecutive images correspond to a single image's 4 channels
    # Initialize an empty list to hold the reshaped images
    reshaped_images = []

    # Iterate over the images and split the channels into a single stack
    for i in range(num_images):
        # Extract the channels for the ith image
        channels = data[i, :, :, :]  # Shape: (C, Y, X)
        
        # We will stack the channels along a new axis to make the final output (C, Y, X) for each image
        reshaped_images.append(channels)

    # Convert list to a numpy array (reshaped to (N * C, Y, X))
    reshaped_data = np.stack(reshaped_images, axis=0)
    reshaped_data = reshaped_data.reshape(-1, height, width)  # Flatten (N, C) into one stack
    print(f"Reshaped data shape: {reshaped_data.shape}")

    # Save the output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tifffile.imwrite(output_file, reshaped_data)
    print(f"Saved reshaped data to {output_file}")

# Example usage
input_tiff = "predictions/jump_cell_painting/n2v.tiff"
output_tiff = "predictions/jump_cell_painting/n2v_sepchannels.tiff"
split_channels_to_single_stack(input_tiff, output_tiff)

import tifffile

input_tiff = "predictions/jump_cell_painting/n2v.tiff"

data = tifffile.imread(input_tiff)
print(f"Loaded data shape: {data.shape}")

print(f"First Data: {data[0]}")
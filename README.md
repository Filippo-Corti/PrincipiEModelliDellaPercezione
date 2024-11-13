# AI4Life N2V, N2V2 and HDN SUMBISSIONS

### Dataset info
````
jump_cell_painting train: train_shape=(413, 4, 540, 540)
jump_cell_painting val: val_shape=(104, 4, 540, 540)
w2s train: train_shape=(96, 3, 512, 512)
w2s val: val_shape=(24, 3, 512, 512)
hagen train: train_shape=(63, 1024, 1024)
hagen val: val_shape=(16, 1024, 1024)
support train: train_shape=(800, 1024, 1024)
support val: val_shape=(201, 1024, 1024)

````
# AI4Life N2V, N2V2 and HDN SUMBISSIONS

### Dataset info
````
jump_cell_painting train: train_shape=(413, 4, 540, 540)
jump_cell_painting val: val_shape=(104, 4, 540, 540)
w2s train: train_shape=(96, 3, 512, 512)
w2s val: val_shape=(24, 3, 512, 512)
hagen train: train_shape=(63, 1024, 1024)
hagen val: val_shape=(16, 1024, 1024)
support train: train_shape=(800, 1024, 1024)
support val: val_shape=(201, 1024, 1024)

````

## Steps to reproduce

## 1. Activate Conda Environment

``` conda env create -f conda.yml ```

``` conda activate n2v ```

``` pip install -r requirements.txt ```

## 2. Download and split data

Dataset urls and save paths are defined in `datasets.yml`.

``` python datasets.py --split --split_ratio=0.8 --seed=1234567890 ```


## 3. Train the N2V/N2V2 models

Use the `train_n2v_careamist.py` script to train each model.

```
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name hagen
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name jump_cell_painting
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name support
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name w2s
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name hagen --use_n2v2
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name jump_cell_painting  --use_n2v2
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name support --use_n2v2
python train_n2v_careamist.py --epochs 400 --batch_size=512 --output_dir models/n2v_n2v2 --dataset_name w2s --use_n2v2

```

Model folders will be called `models/n2v_n2v2/[n2v | n2v2]_<dataset_name>_chwise`. 

## 4. Generate Ground Truths for HDN Noise Models

```bash

python generate_n2v_predictions.py --model_name=n2v --model_ckpt=models/n2v_n2v2/n2v_jump_cell_painting_chwise/checkpoints/last.ckpt --dataset_name=jump_cell_painting
python generate_n2v_predictions.py --model_name=n2v2 --model_ckpt=models/n2v_n2v2/n2v2_jump_cell_painting_chwise/checkpoints/last.ckpt --dataset_name=jump_cell_painting
python generate_n2v_predictions.py --model_name=n2v --model_ckpt=models/n2v_n2v2/n2v_hagen_chwise/checkpoints/last.ckpt --dataset_name=hagen
python generate_n2v_predictions.py --model_name=n2v2 --model_ckpt=models/n2v_n2v2/n2v2_hagen_chwise/checkpoints/last.ckpt --dataset_name=hagen
python generate_n2v_predictions.py --model_name=n2v --model_ckpt=models/n2v_n2v2/n2v_support_chwise/checkpoints/last.ckpt --dataset_name=support
python generate_n2v_predictions.py --model_name=n2v2 --model_ckpt=models/n2v_n2v2/n2v2_support_chwise/checkpoints/last.ckpt --dataset_name=support
python generate_n2v_predictions.py --model_name=n2v --model_ckpt=models/n2v_n2v2/n2v_w2s_chwise/checkpoints/last.ckpt --dataset_name=w2s
python generate_n2v_predictions.py --model_name=n2v2 --model_ckpt=models/n2v_n2v2/n2v2_w2s_chwise/checkpoints/last.ckpt --dataset_name=w2s

```

## 5. Generate Noise Models

```bash
    python generate_noise_model.py --gt_name=n2v --dataset_name=jump_cell_painting
    python generate_noise_model.py --gt_name=n2v2 --dataset_name=jump_cell_painting
    python generate_noise_model.py --gt_name=n2v --dataset_name=hagen
    python generate_noise_model.py --gt_name=n2v2 --dataset_name=hagen
    python generate_noise_model.py --gt_name=n2v --dataset_name=support
    python generate_noise_model.py --gt_name=n2v2 --dataset_name=support
    python generate_noise_model.py --gt_name=n2v --dataset_name=w2s
    python generate_noise_model.py --gt_name=n2v2 --dataset_name=w2s

```

## 6. Train HDN

Change parameters according to noise model and dataset name to use. Remove `--memload_dataset` flag to avoid loading the whole dataset in memory if you don't have enough RAM.

```bash
    python train_hdn.py --dataset_name=hagen --noise_model=n2v --output_root=models/hdn_n2v --memload_dataset --batch_size=256 --virtual_batch=128
```
## Default Folder Tree Structure after running the scripts

```bash
├── models
│   ├── hdn_n2v
│   │   └── hagen
│   ├── hdn_n2v2
│   │   └── hagen
│   └── n2v_n2v2
│       ├── n2v2_hagen_chwise
│       ├── n2v2_jump_cell_painting_chwise
│       ├── n2v2_support_chwise
│       ├── n2v2_w2s_chwise
│       ├── n2v_hagen_chwise
│       ├── n2v_jump_cell_painting_chwise
│       ├── n2v_support_chwise
│       └── n2v_w2s_chwise
├── noise_models
│   ├── hagen
│   │   ├── n2v
│   │   │   ├── GMM.npz
│   │   │   ├── GMM.png
│   │   │   └── histogram.npy
│   │   └── n2v2
│   │       ├── GMM.npz
│   │       ├── GMM.png
│   │       └── histogram.npy
│   ├── jump_cell_painting
│   │   ├── n2v
│   │   │   ├── ...
│   │   └── n2v2
│   │       ├── ...
│   ├── support
│   │   ├── ...
│   └── w2s
│       ├── ...
├── predictions
│   ├── hagen
│   │   ├── n2v2.tiff
│   │   └── n2v.tiff
│   ├── jump_cell_painting
│   │   ├── ...
│   ├── support
│   │   ├── ...
│   └── w2s
│       ├── ...

```
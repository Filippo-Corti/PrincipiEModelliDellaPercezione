from careamics import CAREamist
from datasets import load_split_datasets, load_datasets_yml, iter_tiff_batch
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import argparse



def generate_predictions(output_root:str, model_ckpt: str, model_name: str, dataset_name: str, dataset_yml: str, batch_size: int):

    output_path = os.path.join(output_root, dataset_name, f"{model_name}.tiff")
    
    print(f'Predicting dataset {dataset_name}')
    dset = [d for d in load_datasets_yml(dataset_yml=dataset_yml) if d['name'] == dataset_name][0]
    
    model = CAREamist(model_ckpt, work_dir=os.path.dirname(os.path.dirname(model_ckpt)))

    predictions = []
    i = 0
    for data_batch in iter_tiff_batch(dset['path'], batch_size):
        print(f"Predicting batch of shape {data_batch.shape}, number {i}")
        pred_batch = model.predict(source=data_batch, data_type='array', axes='SCYX' if data_batch.ndim == 4 else 'SYX')
        predictions.append(pred_batch)
        i+=1
        if i >= 100:
            break
    predictions = np.concatenate(predictions, axis=0).squeeze()
    print(f"Saving predictions of shape {predictions.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, predictions)
    print(f"TIFF file saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model and TIFF dataset")
    parser.add_argument('--output_dir', type=str, default="predictions/", help='Output root folder to store predictions. Predictions will be saved in <output_root>/<dataset_name>/<model_name>.tiff.')
    parser.add_argument('--model_name', type=str, help='Name of the model used to generate predition. This is used to name predictions.')
    parser.add_argument('--model_ckpt', type=str, help='Checkpoint to use')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to predict')
    parser.add_argument('--dataset_yml', type=str, help='Dataset yml descriptor', default="datasets.yml")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction. Avoids OOM.')

    args = parser.parse_args()
    generate_predictions(output_root=args.output_dir,
                         model_ckpt=args.model_ckpt,
                         model_name=args.model_name,
                         dataset_name=args.dataset_name,
                         dataset_yml=args.dataset_yml,
                         batch_size=args.batch_size)

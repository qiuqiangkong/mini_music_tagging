import torch
import time
import pickle
import librosa
import numpy as np
from pathlib import Path
import torch.optim as optim
from data.gtzan import Gtzan, CLASSES_NUM
from data.collate import collate_fn
from models.cnn import Cnn
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from train import get_model

from torch.utils.data.sampler import SequentialSampler


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    fold = 0
    batch_size = 16
    num_workers = 16
    device = "cuda"
    filename = Path(__file__).stem

    classes_num = CLASSES_NUM

    root = "/datasets/gtzan"

    # Load checkpoint
    checkpoint_path = Path("checkpoints", "train", model_name, "latest.pth")
    # checkpoint_path = Path("checkpoints", "train", model_name, "step=6000.pth")

    model = get_model(model_name, classes_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    dataset = Gtzan(
        root=root,
        split="test",
        fold=fold,
    )

    sampler = SequentialSampler(dataset)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    pred_ids = []
    target_ids = []

    for data in dataloader:

        segment = torch.Tensor(data["audio"]).to(device)
        target = data["target"]

        with torch.no_grad():
            model.eval()
            output = model(audio=segment)

        pred_ids.append(np.argmax(output.cpu().numpy(), axis=-1))
        target_ids.append(np.argmax(target, axis=-1))

    pred_ids = np.concatenate(pred_ids, axis=0)
    target_ids = np.concatenate(target_ids, axis=0)
        
    accuracy = np.mean(pred_ids == target_ids)

    print(accuracy)

    from IPython import embed; embed(using=False); os._exit(0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    inference(args)
import torch
import torch.nn.functional as F
import lightning as L
import time
import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
from data.gtzan import Gtzan, CLASSES_NUM
from data.collate import collate_fn
from models.cnn import Cnn
from tqdm import tqdm
import museval
import argparse
import random
from torch.utils.data.sampler import SequentialSampler

from train import validate, get_model, bce_loss, play_audio
from data.samplers import DistributedInfiniteSampler


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    fold = 0
    batch_size = 16
    num_workers = 16
    test_step_frequency = 200
    save_step_frequency = 200
    training_steps = 10000
    debug = False
    device = "cuda"
    filename = Path(__file__).stem
    classes_num = CLASSES_NUM

    checkpoints_dir = Path("./checkpoints", filename, model_name)

    root = "/datasets/gtzan"

    # Fabric
    devices_num = torch.cuda.device_count()

    fabric = L.Fabric(accelerator="cuda", devices=devices_num, strategy="ddp")
    fabric.launch()

    # Dataset
    train_dataset = Gtzan(
        root=root,
        split="train",
        fold=fold,
    )

    test_dataset = Gtzan(
        root=root,
        split="test",
        fold=fold,
    )

    # Train sampler
    train_sampler = DistributedInfiniteSampler(dataset_size=len(train_dataset))
    
    # Validate sampler (optional)
    test_sampler = SequentialSampler(test_dataset)


    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name, classes_num)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader, use_distributed_sampler=False)
    test_dataloader = fabric.setup_dataloaders(test_dataloader, use_distributed_sampler=False)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        audio = data["audio"]
        target = data["target"]

        # Play the audio.
        if debug:
            play_audio(mixture, target)

        optimizer.zero_grad()

        model.train()
        output = model(audio=audio)

        loss = bce_loss(output, target)
        fabric.backward(loss)

        optimizer.step()

        if step % 200 == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))

        # Validate
        # if step % test_step_frequency == 0 and fabric.global_rank == 0:
        if step % test_step_frequency == 0:
            accuracy = validate(model, test_dataloader)
            print("Accuracy: {}".format(accuracy))
            
        # Save model
        if step % save_step_frequency == 0 and fabric.global_rank == 0:

            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def validate(model, dataloader):

    pred_ids = []
    target_ids = []
    device = next(model.parameters()).device

    for data in dataloader:

        segment = data["audio"]
        target = data["target"]

        with torch.no_grad():
            model.eval()
            output = model(audio=segment)

        pred_ids.append(np.argmax(output.cpu().numpy(), axis=-1))
        target_ids.append(np.argmax(target.cpu().numpy(), axis=-1))

    pred_ids = np.concatenate(pred_ids, axis=0)
    target_ids = np.concatenate(target_ids, axis=0)
        
    accuracy = np.mean(pred_ids == target_ids)
    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    train(args)
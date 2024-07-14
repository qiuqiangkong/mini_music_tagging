import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import wandb

wandb.require("core")

from data.gtzan import GTZAN
from models.cnn import Cnn


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    test_fold = 0
    sr = 16000
    batch_size = 16
    num_workers = 16
    pin_memory = True
    learning_rate = 1e-4
    test_step_frequency = 200
    save_step_frequency = 200
    training_steps = 10000
    wandb_log = True
    device = "cuda"

    filename = Path(__file__).stem
    classes_num = GTZAN.classes_num

    checkpoints_dir = Path("./checkpoints", filename, model_name)

    root = "/datasets/gtzan"

    if wandb_log:
        wandb.init(project="mini_music_tagging") 

    # Dataset
    train_dataset = GTZAN(
        root=root,
        split="train",
        test_fold=test_fold,
        sr=sr,
    )

    test_dataset = GTZAN(
        root=root,
        split="test",
        test_fold=test_fold,
        sr=sr,
    )

    # Sampler
    train_sampler = InfiniteSampler(train_dataset)
    
    test_sampler = SequentialSampler(test_dataset)
    
    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=1, 
        pin_memory=pin_memory
    )

    # Model
    model = get_model(model_name, classes_num)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        audio = data["audio"].to(device)
        target = data["target"].to(device)

        # Forward
        model.train()
        output = model(audio=audio)

        # Loss
        loss = bce_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        loss.backward()     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        if step % test_step_frequency == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))
            test_acc = validate(model, test_dataloader)
            print("Accuracy: {}".format(test_acc))

            if wandb_log:
                wandb.log(
                    data={"test_acc": test_acc},
                    step=step
                )

        # Save model
        if step % save_step_frequency == 0:
            checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))

        if step == training_steps:
            break


def get_model(model_name, classes_num):
    if model_name == "Cnn":
        return Cnn(classes_num)
    else:
        raise NotImplementedError


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


class InfiniteSampler:
    def __init__(self, dataset):

        self.indexes = list(range(len(dataset)))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


def validate(model, dataloader):

    device = next(model.parameters()).device

    pred_ids = []
    target_ids = []

    for step, data in enumerate(dataloader):
        
        segment = torch.Tensor(data["audio"]).to(device)
        target = torch.Tensor(data["target"]).to(device)

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
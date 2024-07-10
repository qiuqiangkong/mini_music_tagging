import torch
import torch.nn.functional as F
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

    # Sampler
    train_sampler = Sampler(dataset_size=len(train_dataset))
    
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
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):

        # Move data to device
        audio = data["audio"].to(device)
        target = data["target"].to(device)

        # Play the audio
        if debug:
            play_audio(mixture, target)

        # Forward
        model.train()
        output = model(audio=audio)

        # Loss
        loss = bce_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset parameter.grad to 0
        loss.backward()     # Update parameter.grad
        optimizer.step()    # Update parameters based on parameter.grad
        from IPython import embed; embed(using=False); os._exit(0)

        if step % test_step_frequency == 0:
            print("step: {}, loss: {:.3f}".format(step, loss.item()))
            accuracy = validate(model, test_dataloader)
            print("Accuracy: {}".format(accuracy))

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


def play_audio(mixture, target):
    soundfile.write(file="tmp_mixture.wav", data=mixture[0].cpu().numpy().T, samplerate=44100)
    soundfile.write(file="tmp_target.wav", data=target[0].cpu().numpy().T, samplerate=44100)
    from IPython import embed; embed(using=False); os._exit(0)


class Sampler:
    def __init__(self, dataset_size):
        self.indexes = list(range(dataset_size))
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
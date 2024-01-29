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


def train(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    device = "cuda"
    epochs = 200
    checkpoints_dir = Path("./checkpoints", model_name)
    debug = False

    classes_num = CLASSES_NUM
    
    root = "/home/qiuqiangkong/datasets/gtzan"

    # Dataset
    dataset = Gtzan(
        root=root,
        split="train",
    )

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=8, 
        collate_fn=collate_fn,
        num_workers=8, 
        # num_workers=0, 
        pin_memory=True
    )

    # Model
    model = get_model(model_name, classes_num)
    model.to(device)

    # checkpoint_path = Path("checkpoints", model_name, "latest.pth")
    # model.load_state_dict(torch.load(checkpoint_path))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for epoch in range(1, epochs):
        
        for data in tqdm(dataloader):

            audio = data["audio"].to(device)
            target = data["target"].to(device)

            # Play the audio.
            if debug:
                play_audio(mixture, target)

            optimizer.zero_grad()

            model.train()
            output = model(audio=audio)

            loss = bce_loss(output, target)
            loss.backward()

            optimizer.step()

        print(loss)

        # Save model
        if epoch % 2 == 0:
            checkpoint_path = Path(checkpoints_dir, "epoch={}.pth".format(epoch))
            torch.save(model.state_dict(), checkpoint_path)
            print("Save model to {}".format(checkpoint_path))

            checkpoint_path = Path(checkpoints_dir, "latest.pth")
            torch.save(model.state_dict(), Path(checkpoint_path))
            print("Save model to {}".format(checkpoint_path))


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    train(args)
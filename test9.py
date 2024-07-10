
import argparse
import os
import re

import numpy as np
import PIL
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator
import torchvision
from torchvision import transforms


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.fc1 = nn.Linear(64*7*7, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return x


def training_function():
    
    accelerator = Accelerator()

    # Set the seed before splitting the data.
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    download = False

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(root=".", train=True, download=download, transform=transform)
    eval_dataset = torchvision.datasets.MNIST(root=".", train=False, download=download, transform=transform)


    batch_size = 16

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=16)

    model = Cnn()

    model = model.to(accelerator.device)
    
    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0

    # Now we train the model
    for epoch in range(10):
        model.train()
        
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            print("train", step)
            
        model.eval()
        accurate = 0
        num_elems = 0
        for step, batch in enumerate(eval_dataloader):
            
            print("test", step)

if __name__ == "__main__":
    training_function() 
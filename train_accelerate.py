import argparse
from pathlib import Path

import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import wandb

wandb.require("core")

from data.gtzan import GTZAN
from train import InfiniteSampler, bce_loss, get_model, validate


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

    filename = Path(__file__).stem
    classes_num = GTZAN.classes_num

    if wandb_log:
        wandb.init(project="mini_music_tagging") 

    checkpoints_dir = Path("./checkpoints", filename, model_name) 

    root = "/datasets/gtzan"

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

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare for multiprocessing
    accelerator = Accelerator()
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)

    # Create checkpoints directory
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Train
    for step, data in enumerate(tqdm(train_dataloader)):
        
        audio = data["audio"]
        target = data["target"]

        # Forward
        model.train()
        output = model(audio=audio)

        # Loss
        loss = bce_loss(output, target)

        # Optimize
        optimizer.zero_grad()   # Reset all parameter.grad to 0
        accelerator.backward(loss)     # Update all parameter.grad
        optimizer.step()    # Update all parameters based on all parameter.grad

        # Evaluate
        if step % test_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                if accelerator.num_processes == 1:
                    val_model = model
                else:
                    val_model = model.module

                test_acc = validate(val_model, test_dataloader)
                print("Test Accuracy: {}".format(test_acc))

                if wandb_log:
                    wandb.log(
                        data={"test_acc": test_acc},
                        step=step
                    )

        # Save model
        if step % save_step_frequency == 0:

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:

                unwrapped_model = accelerator.unwrap_model(model)

                checkpoint_path = Path(checkpoints_dir, "step={}.pth".format(step))
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print("Save model to {}".format(checkpoint_path))

                checkpoint_path = Path(checkpoints_dir, "latest.pth")
                torch.save(unwrapped_model.state_dict(), Path(checkpoint_path))
                print("Save model to {}".format(checkpoint_path))


        if step == training_steps:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    train(args)
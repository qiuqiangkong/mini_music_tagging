import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data.sampler import SequentialSampler

from data.gtzan import GTZAN
from train import get_model


def inference(args):

    # Arguments
    model_name = args.model_name

    # Default parameters
    test_fold = 0
    sr = 16000
    batch_size = 16
    device = "cuda"
    filename = Path(__file__).stem
    classes_num = GTZAN.classes_num

    root = "/datasets/gtzan"

    # Load checkpoint
    checkpoint_path = Path("checkpoints", "train", model_name, "latest.pth")

    model = get_model(model_name, classes_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    # Test dataset
    test_dataset = GTZAN(
        root=root,
        split="test",
        test_fold=test_fold,
        sr=sr,
    )

    # Test sampler
    test_sampler = SequentialSampler(test_dataset)

    # Dataloader
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=16, 
        pin_memory=True
    )

    pred_ids = []
    target_ids = []

    for data in test_dataloader:

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

    print("Accuracy: {:.3f}".format(accuracy))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Cnn")
    args = parser.parse_args()

    inference(args)
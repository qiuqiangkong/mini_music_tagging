from typing import Dict, Tuple, Union
import re
import os
import torch
from pathlib import Path
import pandas as pd
import random
import librosa
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from data.audio_io import load


class GTZAN(Dataset):
    r"""GTZAN [1] is a music dataset containing 1000 30-second music. 
    GTZAN contains 10 genres. Audios are sampled at 22,050 Hz. Dataset size is 1.3 GB. 

    [1] Tzanetakis, G., et al., Musical genre classification of audio signals. 2002

    The dataset looks like:

        dataset_root (1.3 GB)
        └── genres
            ├── blues (100 files)
            ├── classical (100 files)
            ├── country (100 files)
            ├── disco (100 files)
            ├── hiphop (100 files)
            ├── jazz (100 files)
            ├── metal (100 files)
            ├── pop (100 files)
            ├── reggae (100 files)
            └── rock (100 files)
    """

    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
        "metal", "pop", "reggae", "rock"]

    classes_num = len(labels)
    lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
    ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

    duration = 30.

    def __init__(
        self, 
        root: str = None, 
        split: Union["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        sr: float = 16000,  # Sampling rate
        download: bool = False,
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.sr = sr
        self.audio_samples = int(GTZAN.duration * self.sr)
        
        if not Path(root).exists():
            raise "Please download the GTZAN dataset from http://marsyas.info/index.html (Invalid anymore. Please search a source)"

        self.meta_dict = self.load_meta()
        # E.g., meta_dict = {
        #     "label": ["blues", "disco", ...],
        #     "audio_name": ["blues.00010.au", "disco00005.au", ...],
        #     "audio_path": ["path/blues.00010.au", "path/disco00005.au", ...]
        # }

    def __getitem__(self, index: int) -> Dict:

        audio_path = self.meta_dict["audio_path"][index]
        label = self.meta_dict["label"][index]

        # Load audio
        audio = self.load_audio(path=audio_path)
        # shape: (channels, audio_samples)

        # Load target
        target = self.load_target(label=label)
        # shape: (classes_num,)

        data = {
            "audio_path": str(audio_path),
            "audio": audio,
            "target": target,
            "label": label
        }

        return data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self) -> Dict:
        r"""Load metadata of the GTZAN dataset.
        """

        labels = GTZAN.labels

        meta_dict = {
            "label": [],
            "audio_name": [],
            "audio_path": []
        }

        audios_dir = Path(self.root, "genres")

        for genre in labels:

            audio_names = sorted(os.listdir(Path(audios_dir, genre)))
            # len(audio_names) = 1000

            train_audio_names, test_audio_names = self.split_train_test(audio_names)
            # len(train_audio_names) = 900
            # len(test_audio_names) = 100

            if self.split == "train":
                filtered_audio_names = train_audio_names

            elif self.split == "test":
                filtered_audio_names = test_audio_names

            for audio_name in filtered_audio_names:

                audio_path = Path(audios_dir, genre, audio_name)

                meta_dict["label"].append(genre)
                meta_dict["audio_name"].append(audio_name)
                meta_dict["audio_path"].append(audio_path)

        return meta_dict

    def split_train_test(self, audio_names: list) -> Tuple[list, list]:

        train_audio_names = []
        test_audio_names = []

        test_ids = range(self.test_fold * 10, (self.test_fold + 1) * 10)
        # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]

        for audio_name in audio_names:

            audio_id = int(re.search(r'\d+', audio_name).group())
            # E.g., if audio_name is "blues.00037.au", then audio_id = 37

            if audio_id in test_ids:
                test_audio_names.append(audio_name)

            else:
                train_audio_names.append(audio_name)

        return train_audio_names, test_audio_names

    def load_audio(self, path):

        audio = load(path=path, sr=self.sr)
        # shape: (channels, audio_samples)

        audio = librosa.util.fix_length(data=audio, size=self.audio_samples, axis=-1)
        # shape: (channels, audio_samples)

        return audio

    def load_target(self, label: str) -> np.ndarray:

        classes_num = GTZAN.classes_num
        lb_to_ix = GTZAN.lb_to_ix

        target = np.zeros(classes_num, dtype="float32")
        class_ix = lb_to_ix[label]
        target[class_ix] = 1

        return target


if __name__ == "__main__":

    # Example
    root = "/datasets/gtzan"

    dataset = GTZAN(
        root=root,
        split="train",
        test_fold=0,
    )

    dataloader = torch.utils.data.DataLoader(dataset=dataset)

    for data in dataloader:
        print(data)
        break
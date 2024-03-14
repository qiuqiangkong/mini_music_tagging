import numpy as np
import torch
import librosa


def collate_fn(list_data_dict):
    """Convert list of data to Tensor."""

    data_dict = {}

    for key in list_data_dict[0].keys():
        
        if isinstance(list_data_dict[0][key], np.ndarray):
            data_dict[key] = torch.Tensor(
                np.stack([dd[key] for dd in list_data_dict], axis=0)
            )

        else:
            data_dict[key] = [dd[key] for dd in list_data_dict]

    return data_dict
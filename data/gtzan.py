import re
import os
import torch
from pathlib import Path
import pandas as pd
import random
import librosa
import torchaudio
import numpy as np


LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", 
    "pop", "reggae", "rock"]
LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}
CLASSES_NUM = len(LABELS)


class Gtzan:
	
	def __init__(
		self, 
		root: str = None, 
		split: str = "train",
		fold=0,
	):
	
		self.root = root
		self.split = split
		self.fold = fold
		
		self.sample_rate = 16000
		self.fps = 100

		segment_sec = 30
		self.segment_samples = int(segment_sec * self.sample_rate)
		
		self.meta_dict = self.load_meta()
		self.audios_num = len(self.meta_dict["audio_name"])
				
	def load_meta(self):

		audios_dir = Path(self.root, "genres")

		genres = sorted(os.listdir(audios_dir))

		meta_dict = {
			"label": [],
			"audio_name": [],
			"audio_path": []
		}

		for genre in genres:

			audio_names = sorted(os.listdir(Path(audios_dir, genre)))

			train_audio_names, test_audio_names = self.split_train_test(audio_names)

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

	def split_train_test(self, audio_names):

		# audio_names = sorted(os.listdir(Path(audios_dir, genre)))
		train_audio_names = []
		test_audio_names = []

		test_ids = range(self.fold * 10, (self.fold + 1) * 10)

		for audio_name in audio_names:

			audio_id = int(re.search(r'\d+', audio_name).group())

			if audio_id in test_ids:
				test_audio_names.append(audio_name)
			else:
				train_audio_names.append(audio_name)

		return train_audio_names, test_audio_names
		

	def __getitem__(self, index):

		audio_path = self.meta_dict["audio_path"][index]

		# Load audio.
		audio = self.load_audio(audio_path)
		# shape: (audio_samples)

		label = self.meta_dict["label"][index]
		class_index = LB_TO_IX[label]

		target = np.zeros(CLASSES_NUM)
		target[class_index] = 1

		data = {
			"audio_path": str(audio_path),
			"audio": audio,
			"target": target
		}

		return data

	def __len__(self):

		return self.audios_num

	def load_audio(self, audio_path):

		orig_sr = librosa.get_samplerate(audio_path)

		audio, _ = torchaudio.load(audio_path)
		# (channels, audio_samples)

		audio = torch.mean(audio, dim=0)
		# shape: (audio_samples,)

		audio = torchaudio.functional.resample(
			waveform=audio, 
			orig_freq=orig_sr, 
			new_freq=self.sample_rate
		)
		# shape: (audio_samples,)

		audio = np.array(librosa.util.fix_length(
			data=audio, 
			size=self.segment_samples, 
			axis=0
		))
		# shape: (audio_samples,)

		return audio

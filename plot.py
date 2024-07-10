from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

from data.gtzan import GTZAN


def plot():

	root = "/datasets/gtzan"
	sr = 44100

	labels = GTZAN.labels

	fig, axs = plt.subplots(3, 4, sharex=True, figsize=(10, 4))

	for i, label in enumerate(labels):

		audio_path = Path(root, "genres", label, "{}.00000.au".format(label))

		audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
		
		mel_sp = librosa.feature.melspectrogram(
			y=audio, 
			sr=sr, 
			n_fft=2048, 
			hop_length=441, 
			n_mels=128,
		)
		# (freq_bins, frames_num)

		axs[i // 4, i % 4].matshow(np.log(mel_sp), origin='lower', aspect='auto', cmap='jet')
		axs[i // 4, i % 4].set_title(label)
		axs[i // 4, i % 4].axis('off')

	for i in range(10, 12):
		axs[i // 4, i % 4].set_visible(False)

	plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

	out_path = "data_spectrograms.png"
	plt.savefig(out_path)
	print("Write out to {}".format(out_path))


if __name__ == "__main__":

	plot()
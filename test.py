# %%
import scipy.io as sio
import matplotlib.pyplot as plt

import audiomentations as am
import noisereduce as nr
import librosa

# %%
signal, rate = librosa.load(
    '/src/datasets/base_portuguese/67/p8e672c5ec4aa4b9bb72a757506fd7700_s01_a00.wav')

# %%
reduced_noise = nr.reduce_noise(signal, rate)

sio.wavfile.write('/src/tcc/test_noise_reduced.wav', rate, reduced_noise)

# %%
audioment = am.Compose([
    am.Normalize(p=1)
])

normalized_audio = audioment(reduced_noise, rate)

sio.wavfile.write('/src/tcc/test_noise_reduced_normalized.wav',
                  rate, normalized_audio)

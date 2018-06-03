#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pydub
from pydub import AudioSegment, utils
import os

AudioSegment.converter = pydub.utils.which("ffmpeg")

# data_dir = sys.argv[1]
data_dir = '/home/moniczka/masters/audio/ballet'


def merge_files():
    folders = os.listdir(data_dir)

    for folder in folders:
        combined = AudioSegment.empty()
        for song in os.listdir(os.path.join(data_dir, folder)):
            print(utils.mediainfo(os.path.join(data_dir, folder, song)))
            print(os.path.join(data_dir, folder, song))
            combined += AudioSegment(os.path.join(data_dir, folder, song))
            # hm = AudioSegment(os.path.join(data_dir, folder, song), 'mp3')
            combined += os.path.join(data_dir, folder, song)
        combined.export(os.path.join(data_dir, folder + '_full'), format="mp3")


y, sr = librosa.load(os.path.join(data_dir, '0cTtPWmpGdc_023.webm'))
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
print(tempogram)
# Compute global onset autocorrelation
ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
ac_global = librosa.util.normalize(ac_global)
# Estimate the global tempo for display purposes
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
print(tempo)

plt.figure(figsize=(8, 8))
plt.subplot(4, 1, 1)
plt.plot(oenv, label='Onset strength')
plt.xticks([])
plt.legend(frameon=True)
plt.axis('tight')
plt.subplot(4, 1, 2)
# We'll truncate the display to a narrower range of tempi
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
plt.axhline(tempo, color='w', linestyle='--', alpha=1, label='Estimated tempo={:g}'.format(tempo))
plt.legend(frameon=True, framealpha=0.75)
plt.subplot(4, 1, 3)
x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr, num=tempogram.shape[0])
plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
plt.xlabel('Lag (seconds)')
plt.axis('tight')
plt.legend(frameon=True)
plt.subplot(4, 1, 4)
# We can also plot on a BPM axis
freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
             label='Mean local autocorrelation', basex=2)
plt.semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
             label='Global autocorrelation', basex=2)
plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
            label='Estimated tempo={:g}'.format(tempo))
plt.legend(frameon=True)
plt.xlabel('BPM')
plt.axis('tight')
plt.grid()
plt.tight_layout()
plt.show()
merge_files()

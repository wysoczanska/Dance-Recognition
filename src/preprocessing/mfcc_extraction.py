import argparse
import multiprocessing as mp

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math


def generate_histogram(sample, root_dir, out_dir):
    y, sr = librosa.load(os.path.join(root_dir, sample.target, sample.filename[:-4] + '.wav'))
    hop_length = 512
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    log_S = librosa.power_to_db(S, ref=np.max)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Let's draw transparent lines over the beat frames
    plt.vlines(librosa.frames_to_time(beats),
               1, 0.5 * sr,
               colors='w', linestyles='-', linewidth=2, alpha=0.5)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    splits = librosa.util.frame(y, frame_length=int(sr / 2), hop_length=int(sr / 2))
    if not os.path.exists(os.path.join(out_dir, sample.target)):
        os.mkdir(os.path.join(out_dir, sample.target))
    # oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,hop_length = hop_length)
    # librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time',y_axis='tempo')
    # plt.draw()
    # tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length = hop_length)[0]
    # print(tempo)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_name = sample.filename[:-4] + '.png'
    # plt.set_cmap('inferno')
    plt.savefig(os.path.join(out_dir, sample.target, file_name), dpi=100, bbox_inches=0)
    plt.close()
    print(sample.target + '/' + file_name)

    # for i in range(splits.shape[-1]):
    #     print(splits.shape)
    #     oenv = librosa.onset.onset_strength(y=splits[:,i], sr=sr, hop_length=hop_length)
    #
    #     tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    #     librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
    #     plt.draw()
    #     plt.set_cmap('inferno')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     file_name = sample.filename[:-4] + '_%02d.png' % i
    #     plt.savefig(os.path.join(out_dir, sample.target, file_name), dpi=100, bbox_inches=0)
    #     plt.close()
    #     print(sample.target + '/' + file_name)

def generate_spectrogram(sample, root_dir, out_dir):

    # start_time = int(sample.filename[-2:])+int(sample.filename[-3])*60
    y, sr = librosa.load(os.path.join(root_dir, sample.target, sample.filename[:-4] + '.wav'), duration=8)
    # splits = librosa.util.frame(y, frame_length=int(sr/2), hop_length=512)
    if not os.path.exists(os.path.join(out_dir, sample.target)):
        os.mkdir(os.path.join(out_dir, sample.target))
    # librosa.output.write_wav(os.path.join(out_dir, sample.target, sample.filename[:-4] + '.wav'), y, sr)
    figsize = 1.76, 1.28
    plt.rcParams["figure.figsize"] = figsize

    S = librosa.feature.melspectrogram(y=y, n_mels=128,  hop_length=1024)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    dpi = 100
    # What size does the figure need to be in inches to fit the image?

    plt.draw()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_name = sample.filename[:-4] + '.png'

    plt.savefig(os.path.join(out_dir, sample.target, file_name), dpi=dpi, bbox_inches=0)
    plt.close()
    print(sample.target + '/' + file_name)

    # for i in range(splits.shape[-1]):
    #     S = librosa.feature.melspectrogram(y=splits[:,i], sr=sr, n_mels=96)
    #     librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    #     plt.draw()
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     file_name = sample.filename[:-4] + '_%02d.png' % i
    #     plt.savefig(os.path.join(out_dir, sample.target, file_name), dpi=100, bbox_inches=0)
    #     plt.close()
    #     print(sample.target + '/' + file_name)


def main(root_dir, out_dir, durations_df_dir, type_features):
    durations_df=pd.read_csv(durations_df_dir, sep='\t', names=['target', 'filename', 'duration'])
    if type_features == 'mfcc':
        durations_df.apply(generate_spectrogram, args=(root_dir, out_dir), axis=1)
    else:
        durations_df.apply(generate_histogram, args=(root_dir, out_dir), axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        help="root directory for the whole data set's txt poses files")
    parser.add_argument('--output_dir',
                        help='root directory for output images')
    parser.add_argument('--csv_file',
                        help='CSV file with duration')
    parser.add_argument('--features', choices=('beat', 'mfcc'))

    args = parser.parse_args()
    main(args.root_dir, args.output_dir, args.csv_file, args.features)
    # output = mp.Queue()
    # classes = ['rumba', 'flamenco']
    # processes = [mp.Process(target=main, args=(args.root_dir, args.output_dir, classes[x])) for x in range(2)]
    #
    # # Run processes
    # for p in processes:
    #     p.start()
    #
    # # Exit the completed processes
    # for p in processes:
    #     p.join()
    #
    # # Get process results from the output queue
    # results = [output.get() for p in processes]
    #
    # print(results)

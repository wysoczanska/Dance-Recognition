import argparse
import multiprocessing as mp

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math


def generate_spectrogram(sample, root_dir, out_dir):

    start_time = int(sample.filename[-2:])+int(sample.filename[-3])*60
    y, sr = librosa.load(os.path.join(root_dir, sample.target, sample.filename[:-4] + '.wav'), offset=start_time,
                         duration=sample.duration/30)
    splits = librosa.util.frame(y, frame_length=int(sr/2), hop_length=int(sr/2))
    print(int(sr/2))
    if not os.path.exists(os.path.join(out_dir, sample.target)):
        os.mkdir(os.path.join(out_dir, sample.target))
    for i in range(splits.shape[-1]):
        S = librosa.feature.melspectrogram(y=splits[:,i], sr=sr, n_mels=96)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
        plt.draw()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        file_name = sample.filename[:-4] + '_%02d.png' % i
        plt.savefig(os.path.join(out_dir, sample.target, file_name), dpi=100, bbox_inches=0)
        plt.close()
        print(sample.target + '/' + file_name)


def main(root_dir, out_dir, durations_df_dir):
    durations_df=pd.read_csv(durations_df_dir, sep='\t', names=['target', 'filename', 'duration'])
    durations_df.apply(generate_spectrogram, args=(root_dir, out_dir), axis=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        help="root directory for the whole data set's txt poses files")
    parser.add_argument('--output_dir',
                        help='root directory for output images')
    parser.add_argument('--csv_file',
                        help='CSV file with duration')

    args = parser.parse_args()
    main(args.root_dir, args.output_dir, args.csv_file)
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

import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_spectrogram(filename, root_dir, out_dir):
    y, sr = librosa.load(os.path.join(root_dir,filename))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    f = plt.figure()
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    plt.draw()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(out_dir, filename[:-4] + '.png'), dpi=100, bbox_inches=0)


def main(root_dir, out_dir):
    classes = os.listdir(root_dir)
    for cls in classes:
        out_dir = os.path.join(out_dir, cls)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        audio_files = os.listdir(os.path.join(root_dir, cls))
        for file in audio_files:
            generate_spectrogram(file, os.path.join(root_dir, cls), out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        help="root directory for the whole data set's txt poses files")
    parser.add_argument('--output_dir',
                        help='root directory for output images')

    args = parser.parse_args()
    main(args.root_dir, args.output_dir)

import os
from collections import Counter
import sys
import glob
import argparse
from pipes import quote
from multiprocessing import Pool, current_process


def check_directories(path, rgb_dir='letsdance_split', flow_dir='flow', skeleton_dir='skeleton'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))

    split='train'
    classes = os.listdir(os.path.join(path, flow_dir))

    white_list=[]
    black_list=[]

    with open('train_clips.txt', 'w') as fl:
        for cls in classes:
            if os.path.isdir(os.path.join(path, rgb_dir, split, cls)):
                rgb_files = os.listdir(os.path.join(path, rgb_dir, split, cls))
                flow_files = os.listdir(os.path.join(path, flow_dir, cls))
                skeleton_files = os.listdir(os.path.join(path, skeleton_dir, cls))

                rgb_counts, flow_counts, skeleton_counts = Counter([file[:10] for file in rgb_files]), \
                                                           Counter([file[:10] for file in flow_files]), \
                                                           Counter([file[:10] for file in skeleton_files])
                print(flow_counts)


                for clip, duration in rgb_counts.items():
                    print(clip)

                    if duration == flow_counts[clip] & duration == skeleton_counts[clip]==300:
                        fl.write(clip+'   '+cls+' '+str(duration)+'\n')

if __name__ == '__main__':

    check_directories('/home/mwysocz1/letsdance')

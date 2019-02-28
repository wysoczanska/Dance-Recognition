from collections import Counter

import os

# assuming these subfolders names
rgb_dir = 'letsdance_split'
flow_dir = 'flow'
skeleton_dir = 'skeleton'


def check_directories(path, split):
    """
    Parse directories holding extracted frames from standard benchmarks
    """

    classes = os.listdir(os.path.join(path, rgb_dir, split))
    white_list=[]

    for cls in classes:
            if os.path.isdir(os.path.join(path, rgb_dir, split, cls)):
                rgb_files = os.listdir(os.path.join(path, rgb_dir, split, cls))
                flow_files = os.listdir(os.path.join(path, flow_dir, cls))

                skeleton_files = os.listdir(os.path.join(path, skeleton_dir, cls))

                rgb_counts, flow_counts, skeleton_counts = Counter([file[:10] for file in rgb_files]), \
                                                           Counter([file[:10] for file in flow_files]), \
                                                           Counter([file[:10] for file in skeleton_files])

                for clip, duration in rgb_counts.items():
                    cnt =0
                    if clip in flow_counts.keys():
                        if (duration == flow_counts[clip]) and (duration == skeleton_counts[clip] == 300):
                            white_list.append(clip)



    return white_list


def build_complete_files_list(path, split):

    with open(split+'_complete'+'.txt', 'w') as fl:
        complete_clips = check_directories(path, split)
        classes = os.listdir(os.path.join(path, flow_dir))

        if 'HJyQjQy9n-' in complete_clips: print('True')
        print(complete_clips)

        for cls in classes:
            if os.path.isdir(os.path.join(path, rgb_dir, split, cls)):
                rgb_files = os.listdir(os.path.join(path, rgb_dir, split, cls))
                for rgb_file in rgb_files:
                    if rgb_file[:10] in complete_clips:
                        fl.write(os.path.join(split, cls, rgb_file)+'   ' + cls + ' ' + str(300) + '\n')



if __name__ == '__main__':

    build_complete_files_list('/home/mwysocz1/letsdance', 'train')

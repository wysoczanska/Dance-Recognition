from collections import Counter

import os

path = '/data/trackml/letsdance/audio'
all_classes = os.listdir(path)


with open('audios_list.txt', 'w') as build_f:
    for cls in all_classes:
            files = os.listdir(os.path.join(path, cls))
            rgb_counts = Counter([file[:15] for file in files])
            for clip, duration in rgb_counts.items():
                build_f.write(cls + '\t' + clip + '\t' + str(duration) + '\n')




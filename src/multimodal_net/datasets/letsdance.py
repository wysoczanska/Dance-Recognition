import math
import sys
import cv2
import numpy as np
import os
import torch.utils.data as data_flow
import skorch
import pickle

rgb_dir = 'rgb/rgb'
flow_dir = 'flow_png'
skeleton_dir = 'densepose/rgb'
audio_dir = 'audio_mfcc_new'


def find_classes(dir):
    c_dir = os.path.join(dir, rgb_dir)
    classes = [d for d in os.listdir(c_dir) if os.path.isdir(os.path.join(c_dir, d))]
    classes.sort()
    print(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(source, seg_length, audio=False):

    if not os.path.exists(source):
        print("Setting file %s for Lets dance dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                duration = int(line_info[2])
                num_segments = int(math.floor(duration/seg_length))
                if audio:
                    target = line_info[0]
                    file_name = line_info[1]
                    print(file_name)
                    item = (file_name, target)
                else:
                    for seg_id in range(0, num_segments):
                        init_frame_id = seg_id * seg_length + 1
                        target = line_info[0]
                        item = (line_info[1], init_frame_id, target)
                        clips.append(item)
    return clips


def read_segment(clip_id, init_frame_id, root, target, seg_length, is_color, name_pattern):

    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0

    sampled_list_rgb = []
    sampled_list_flow = []
    sampled_list_skeleton = []
    rgb_extension = '.jpg'
    rest_extension = '.png'

    for length_id in range(seg_length):
        frame_id = name_pattern % (clip_id, init_frame_id + length_id)

        rgb_path = os.path.join(root, rgb_dir, target, frame_id + rgb_extension)
        flow_path = os.path.join(root, flow_dir, target, frame_id + rest_extension)
        skeleton_path = os.path.join(root, skeleton_dir, target, frame_id + rest_extension)
        cv_img_origin = cv2.imread(rgb_path, cv_read_flag)
        cv_img_flow = cv2.imread(flow_path, cv_read_flag)
        cv_img_skeleton = cv2.imread(skeleton_path, cv_read_flag)

        if cv_img_origin is None or cv_img_flow is None or cv_img_skeleton is None:
            print("Could not load file %s, %s" % (target, frame_id))
        cv_img_rgb = cv2.cvtColor(cv_img_origin, cv2.COLOR_BGR2RGB)
        cv_img_flow = cv2.cvtColor(cv_img_flow, cv2.COLOR_BGR2RGB)
        cv_img_skeleton = cv2.cvtColor(cv_img_skeleton, cv2.COLOR_BGR2RGB)

        sampled_list_rgb.append(cv_img_rgb), sampled_list_flow.append(cv_img_flow), \
        sampled_list_skeleton.append(cv_img_skeleton)

    clip_input = {'rgb': np.concatenate(sampled_list_rgb, axis=2), 'flow': np.concatenate(sampled_list_flow, axis=2),
                  'skeleton': np.concatenate(sampled_list_skeleton, axis=2)}
    return clip_input


class Letsdance(data_flow.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 is_color=True,
                 new_length=1,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 return_id=False
                 ):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(source, new_length)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        self.root = root
        self.source = source
        self.phase = phase
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.is_color = is_color
        self.name_pattern = "%s_%04d"
        self.seg_length = new_length
        self.return_id = return_id

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        clip_id, init_frame_id, cls = self.clips[index]
        target = self.class_to_idx[cls]
        clip_input = read_segment(clip_id, init_frame_id, self.root, cls, self.seg_length, self.is_color, self.name_pattern)

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        if self.return_id:
            return clip_input, target, clip_id
        else:
            return clip_input, target

    def __len__(self):
        return len(self.clips)


class Letsdance_audio(skorch.dataset.Dataset):
    def __init__(self,
                 root,
                 source,
                 phase,
                 is_color=True,
                 new_length=1,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 return_id=False
                 ):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(source, new_length, audio=True)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        self.root = root
        self.source = source
        self.phase = phase
        self.classes = classes
        self.class_to_idx = class_to_idx
        pickle.dump(class_to_idx, open('classes_to_idx.pkl', 'wb'))
        self.clips = clips
        self.is_color = is_color
        self.name_pattern = "%s.png"
        self.seg_length = new_length
        self.return_id = return_id


        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        clip_id, cls = self.clips[index]
        target = self.class_to_idx[cls]
        file_name = self.name_pattern % clip_id

        clip_input = cv2.imread(os.path.join(self.root, audio_dir, cls, file_name), cv2.IMREAD_COLOR)
        if clip_input is None:
            print("Could not load file %s, %s" % (target, file_name))
        sample = cv2.cvtColor(clip_input, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_id:
            return sample, target, clip_id
        else:
            return sample, target


    def __len__(self):
        return len(self.clips)


class ModalityInputIterator(object):
    def __init__(self, batch_size, root, file_list, new_length, extension):
        self.batch_size = batch_size
        self.root = root
        self.modal_dirs = {'rgb': rgb_dir, 'flow': flow_dir, 'skeleton': skeleton_dir}
        self.source = file_list
        clips = make_dataset(self.source, new_length)

        self.clips = clips
        self.name_pattern = "%s_%04d"+extension
        self.seg_length = new_length

    def __iter__(self):
        self.i = 0
        self.n = len(self.clips)
        return self

    def __next__(self):
        batch = {}
        labels = []
        for _ in range(self.batch_size):
            for modality, dir in self.modal_dirs.items():
                batch[modality] = []
                for l_id in range(self.seg_length):
                    clip_id, init_frame_id, label = self.clips[self.i]
                    file_name = self.name_pattern % (clip_id, init_frame_id + l_id)
                    f = open(os.path.join(self.root, dir, label, file_name), 'rb')
                    batch[modality].append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return batch, labels

    next = __next__

# if __name__ == '__main__':
#     data_flow= '/data/inputs/jersey_number_recognition/letsdance/'
#     train_split_file='/home/m.wysoczanska/Dance-Recognition/src/multimodal_net/datasets/letsdance_splits/train.csv'
#     train_transform = transforms.Compose([
#             # video_transforms.Scale((256)),
#             transforms.Resize((299, 299)),
#             # video_transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # normalize,
#         ])
#     train_dataset = Letsdance(data_flow, source=train_split_file,
#                               phase="train",
#                               is_color=True,
#                               new_length=1,
#                               video_transform=train_transform)
#     mean_flow = 0.
#     std_flow = 0.
#     mean_s = 0.
#     std_s = 0.
#     nb_samples = 0.
#     loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=120,
#         num_workers=30,
#         shuffle=False
#     )
#     print(len(loader))
#     for i, (data, target) in tqdm.tqdm(enumerate(loader)):
#         data_flow=data['flow']
#         data_skeleton=data['skeleton']
#         batch_samples = data_flow.size(0)
#         data_flow = data_flow.view(batch_samples, data_flow.size(1), -1)
#         mean_flow += data_flow.mean(2).sum(0)
#         std_flow += data_flow.std(2).sum(0)
#         mean_s += data_skeleton.mean(2).sum(0)
#         std_s += data_skeleton.std(2).sum(0)
#         nb_samples += batch_samples
#
#     mean_flow /= nb_samples
#     std_flow /= nb_samples
#
#     mean_s /= nb_samples
#     std_s /= nb_samples
#
#     print(mean_flow)
#     print(std_flow)
#
#     print(mean_s)
#     print(std_s)
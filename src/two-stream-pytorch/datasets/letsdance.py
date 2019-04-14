import math
import sys

import cv2
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import torch.utils.data as data
from nvidia.dali.pipeline import Pipeline

rgb_dir = 'rgb/rgb'
flow_dir = 'flow_png'
skeleton_dir = 'densepose/rgb'

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
                if audio:
                    target = line_info[0]
                    item = (line_info[1], target)
                    clips.append(item)
                else:
                    duration = int(line_info[2])
                    num_segments = int(math.floor(duration/seg_length))
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

        # if not os.path.isfile(rgb_path):
        #     print("Could not load file rgb %s, %s" % (target, frame_id))
        #     sys.exit()
        #
        # if not os.path.isfile(flow_path):
        #     print("Could not load file flow %s, %s" % (target, frame_id))
        #     sys.exit()
        #
        # if not os.path.isfile(skeleton_path):
        #     print("Could not load file skeleton %s, %s" % (target, frame_id))
        #     sys.exit()
    #
        cv_img_origin = cv2.imread(rgb_path, cv_read_flag)
        cv_img_flow = cv2.imread(flow_path, cv_read_flag)
        cv_img_skeleton = cv2.imread(skeleton_path, cv_read_flag)

        if cv_img_origin is None or cv_img_flow is None or cv_img_skeleton is None:
            print("Could not load file %s, %s" % (target, frame_id))
            sys.exit()
        cv_img_rgb = cv2.cvtColor(cv_img_origin, cv2.COLOR_BGR2RGB)
        cv_img_flow = cv2.cvtColor(cv_img_flow, cv2.COLOR_BGR2RGB)
        cv_img_skeleton = cv2.cvtColor(cv_img_skeleton, cv2.COLOR_BGR2RGB)

        sampled_list_rgb.append(cv_img_rgb), sampled_list_flow.append(cv_img_flow), \
        sampled_list_skeleton.append(cv_img_skeleton)

    clip_input = {'rgb': np.concatenate(sampled_list_rgb, axis=2), 'flow': np.concatenate(sampled_list_flow, axis=2),
                  'skeleton': np.concatenate(sampled_list_skeleton, axis=2)}

    return clip_input


class Letsdance(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 is_color=True,
                 new_length=1,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

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

        return clip_input, target

    def __len__(self):
        return len(self.clips)

class Letsdance_audio(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 is_color=True,
                 new_length=1,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

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
        self.clips = clips
        self.is_color = is_color
        self.name_pattern = "%s.png"
        self.seg_length = new_length

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        clip_id, cls = self.clips[index]
        target = self.class_to_idx[cls]
        file_name = self.name_pattern % (clip_id[:-4])

        clip_input = cv2.imread(os.path.join(self.root, 'audio_png', cls, file_name), cv2.IMREAD_COLOR)
        if clip_input is None:
            print("Could not load file %s, %s" % (target, file_name))
            clip_input = np.zeros((224,224, 3))
        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target

    def __len__(self):
        return len(self.clips)

class ModalityInputIterator(object):
    def __init__(self, batch_size, file_list, new_length, extension):
        self.batch_size = batch_size
        self.root = file_list
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
        batch = []
        labels = []
        for _ in range(self.batch_size):
            for l_id in range(self.seg_length):
                clip_id, init_frame_id, label = self.clips[self.i]
                file_name = self.name_pattern % (clip_id, init_frame_id + l_id)
                f = open(os.path.join(self.root, label, file_name), 'rb')
                batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return batch, labels

    next = __next__


class LetsDancePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(LetsDancePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.input = ops.ExternalSource()
        self.label = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)

    def define_graph(self):
        self.jpegs = self.input()
        images = self.decode(self.jpegs)
        self.flow_jpegs = self.flow()
        self.skeleton_jpegs = self.skeleton()
        self.labels = self.input_label()
        images_rgb = self.decode(self.jpegs)
        images_flow = self.decode(self.jpegs)
        images_flow = self.decode(self.jpegs)
        output = self.cast(images)
        return (output, self.labels)

    # def iter_setup(self):
    #     (images, labels) = iterator.next()
    #     self.feed_input(self.jpegs, images)
    #     self.feed_input(self.labels, labels)

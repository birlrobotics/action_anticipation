# To avoid error if ROS is installed in the device
import sys
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import os   
import cv2
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import utils.io as io
from dataset.config import BF_CONFIG, BF_ACTION_CLASS

_DatasetPath = BF_CONFIG["data_dir"]
_ALLFILES = ['P'+str(i).zfill(2) for i in range(3, 55)]

class BreakfastDataset(Dataset):
    # mode = ["trainval", "train", "val"]
    def __init__(self, mode='train', split_idx=0, preproc=None, over_write=False):
        super(BreakfastDataset, self).__init__()
        # prepare dataset
        # get original notation information
        self.notation_info = io.loads_json(os.path.join(_DatasetPath, "notation.json"))
        if not over_write and os.path.exists(os.path.join(_DatasetPath, "split" + str(split_idx) + '.json')):
            all_data = io.loads_json(os.path.join(_DatasetPath, "split" + str(split_idx) + '.json'))
            self.data_dir = all_data[mode].copy()
            del all_data
        else:
            self._split_data(mode, split_idx, BF_CONFIG["test_size"])
        
        self.video_len = BF_CONFIG['video_len']
        self.sample_num = BF_CONFIG['sample_num_each_clip']
        self.load_mode = BF_CONFIG['load_mode']

    def _split_data(self, mode, split_idx, test_size):
        # construct splited examples
        split_data = {"trainval":[], "train":[], "val":[], "test":[]}
        split_file = open(os.path.join(_DatasetPath, 'split.txt'))
        split_info = split_file.readlines()
        test_examples_ival = split_info[split_idx].rstrip().split('-')
        test_examples = ['P'+str(i).zfill(2) for i in range(int(test_examples_ival[0][1:]), int(test_examples_ival[1][1:])+1)]
        examples = [i for i in _ALLFILES if i not in test_examples]
        for i in examples:
            for j in self.notation_info.keys():
                if i in j:
                    split_data["trainval"].append(j)
        split_data["test"] = [j for j in self.notation_info if j not in split_data["trainval"]]
        split_data["train"], split_data["val"] = train_test_split(split_data["trainval"], test_size=test_size, random_state=42)
        io.dumps_json(split_data, os.path.join(_DatasetPath, "split" + str(split_idx) + '.json'))
        self.data_dir = split_data[mode].copy()
        del split_data

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        # load data: [len, sample_num, H, W, C]
        clips, labels, pad_num = self._load_clips_and_labels(self.data_dir[index], load_mode=self.load_mode)
        # TODO: Add preprocess here
        clips = self._normalize(clips)
        # [len, sample_num, H, W, C] --> [len, C, sample_num, H, W]
        clips = clips.transpose((0, 4, 1, 2, 3))
        
        return torch.from_numpy(clips), torch.from_numpy(labels), pad_num

    def _load_clips_and_labels(self, data_dir, load_mode='all'):
        frame_path = data_dir
        # # for baxter workstation
        # if any([i in frame_path for i in ["P42", "P43", "P44", "P45", "P46", "P47", "P48", "P49", "P50", "P51", "P52", "P53", "P54"]]):
        #     frame_path = frame_path.replace("./dataset/breakfast/rgb_frame", "../supp_dataset/breakfast")
        frames = sorted([os.path.join(frame_path, img) for img in os.listdir(frame_path)])
        if load_mode == 'all':
            buffer, labels = self._load_clips_all(data_dir, frames, random_sample=False)
            return buffer, labels
        if load_mode == 'uniform':
            buffer, labels, pad_num = self._load_clips_uniformly(data_dir, frames)
            return buffer, labels, pad_num
    
    def _load_clips_uniformly(self, data_dir, frames):
        ori_video_len = len(frames) // BF_CONFIG['FPS']
        pad_num = 0
        if ori_video_len <= self.video_len:
            buffer, labels = self._sample(data_dir, frames, 1)
        else:
            sample_ratio = self.video_len / ori_video_len
            buffer, labels = self._sample(data_dir, frames, sample_ratio)
        if buffer.shape[0] < self.video_len:
            pad_num = self.video_len-buffer.shape[0]   
            buffer = np.concatenate((buffer, np.tile(buffer[-1][None,:], (pad_num, 1, 1, 1, 1))))
            # labels = np.concatenate((labels, np.tile(labels[-1][None,:], (self.video_len-labels.shape[0], 1))))
            labels = np.concatenate((labels, np.array([-100]*pad_num)))
        return buffer, labels, pad_num
    
    def _sample(self, data_dir, frames, sample_ratio=1):
        labels_list = []
        buffer_list = []
        # sample the frames within each segment
        for k, v in sorted(self.notation_info[data_dir].items(), key=lambda a: int(a[0].split('-')[0])):
            # temp_label = np.zeros(len(BF_ACTION_CLASS), np.dtype('float32'))
            a, b= k.split('-')
            # skip the extremely short segments
            if a==b or (int(b)-int(a)<8):
                continue
            clip_num = (int(b)-int(a))*sample_ratio // 15 if (int(b)-int(a))*sample_ratio // 15 else 1
            try:
                for s in sorted(random.sample((range(int(a), int(b), 15)), int(clip_num))):
                    temp_data = np.empty((self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
                    # to ensure each segment have at least one clip
                    if clip_num == 1:
                        if int(b) > len(frames):
                            break
                        s = 1 if int(b)-8==int(a) else random.choice(range(int(a), int(b)-8))
                        for i, j in enumerate(range(s, s+8)):
                            frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                            temp_data[i] = frame   
                        # temp_label[BF_ACTION_CLASS.index(v)] = 1 
                        # labels_list.append(temp_label)
                        buffer_list.append(temp_data)
                        labels_list.append(BF_ACTION_CLASS.index(v))
                        break   
                    # set "flag" to avoid the empty temp_data                 
                    flag = False
                    if s+15 <= int(b):
                        if s+15 > len(frames):
                            break
                        flag = True
                        for i, j in enumerate(range(s, s+15, 2)):
                            frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                            temp_data[i] = frame
                        # temp_label[BF_ACTION_CLASS.index(v)] = 1
                        temp_label = BF_ACTION_CLASS.index(v) 
                    elif s+8 <= int(b):
                        if s+8 > len(frames):
                            break
                        flag = True
                        for i, j in enumerate(range(int(s), int(s)+8)):
                            frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                            temp_data[i] = frame
                        # temp_label[BF_ACTION_CLASS.index(v)] = 1
                        temp_label = BF_ACTION_CLASS.index(v)
                    if flag:
                        buffer_list.append(temp_data)
                        labels_list.append(temp_label)
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print(e)
        buffer = np.array(buffer_list)
        labels = np.array(labels_list)
        return buffer, labels

    def _load_clips_all(self, data_dir, frames, random_sample=False):
        ori_video_len = len(frames) // BF_CONFIG['FPS']
        labels = np.zeros((ori_video_len, len(BF_ACTION_CLASS)), np.dtype('float32'))
        buffer = np.empty((ori_video_len, self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
        temp = np.empty((self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
        # TODO: optimize this part
        for i in range(ori_video_len):
            f4_idx = 0; f5_idx = 0
            if random_sample:
                for idx, j in enumerate(sorted(random.sample(range(i*15, (i+1)*15), 8))):
                     # read image and convert it from BGR to RGB
                    frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                    temp[idx] = frame        
                    if idx == 3: f4_idx = j 
                    if idx == 4: f5_idx = j           
            else:
                for idx, j in enumerate(range(i*15, (i+1)*15, 2)):
                    frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                    temp[idx] = frame
                    if idx == 3: f4_idx = j 
                    if idx == 4: f5_idx = j 
            # get the label for each clip
            f4_label = None; f5_label = None
            for k, v in self.notation_info[data_dir].items():
                if f4_label and f5_label:
                    break
                a, b = k.split('-')
                if f4_idx+1 in range(int(a), int(b)):
                    f4_label = v
                if f5_idx+1 in range(int(a), int(b)):
                    f5_label = v
            labels[i][BF_ACTION_CLASS.index(f5_label if f4_label == f5_label else f4_label)] = 1
            buffer[i] = temp
        return buffer, labels

    def _normalize(self, buffer):
        means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # normalization
        buffer /= 255.0
        # buffer = (buffer - means[None, None, None, None, :]) / stds[None, None, None, None, :]
        buffer = (buffer - means) / stds
        return buffer
        
def collect_fn(batch):
    pass


if __name__ == "__main__":
    dataset = BreakfastDataset()
    for data in iter(dataset):
        print(data)
    print("test finished!!!")

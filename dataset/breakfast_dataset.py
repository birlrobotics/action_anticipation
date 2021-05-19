# To avoid error if ROS is installed in the device
import sys
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import os
import cv2
import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import utils.io as io
from dataset.config import BF_CONFIG, BF_ACTION_CLASS

_DatasetPath = BF_CONFIG["data_dir"]
_ALLFILES = ['P'+str(i).zfill(2) for i in range(3, 55)]           

class BreakfastDataset(Dataset):
    '''    
        Dataset class for breakfast:
        Args:
            mode = ["trainval", "train", "val", 'test']
            task = ["recog_only", "recog_anti"]
            feat_type = ["online", "offline"]
            data_type: 'all' means we create all types of data with three obs_prec ([0.2, 0.3, 0.5]) in one time 
                        so that we can shuffle the data from different videos instead of using the three serial data of a video.
            anti_feat: return the anticipation features or not.  
    '''
    def __init__(self, mode='train', split_idx=0, task='recog_anti', feat_type='offline', data_type='all', anti_feat=False, preproc=None, over_write=False, evaluation=False, data_aug=True):
        super(BreakfastDataset, self).__init__()
        self.video_len = BF_CONFIG['video_len']
        self.sample_num = BF_CONFIG['sample_num_each_clip']
        self.load_mode = BF_CONFIG['load_mode']
        self.task = task
        self.anti_feat = anti_feat
        self.feat_type = feat_type
        self.data_type = data_type
        self.data_aug = data_aug
        # get original notation information
        self.notation_info = io.loads_json(os.path.join(_DatasetPath, "notation.json"))
        if feat_type == "offline":
            self.data_feat = h5py.File(os.path.join(BF_CONFIG["data_dir"], BF_CONFIG["feat_hdf5_name"]), 'r')
        
        if not over_write and os.path.exists(os.path.join(_DatasetPath, "split" + str(split_idx) + '.json')):
            all_data_dir = io.loads_json(os.path.join(_DatasetPath, "split" + str(split_idx) + '.json'))
            self.data_dir = all_data_dir[mode].copy()
            del all_data_dir
        else:
            self._split_data(mode, split_idx, BF_CONFIG["test_size"])

        if data_type == "all" and not evaluation:
            self.all_data = self._recog_anti_all_data_gen(self.data_dir)
            del self.data_feat

        # _, self.data_dir = train_test_split(self.data_dir, test_size=BF_CONFIG["test_size"], random_state=42)

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
        # record the status after segmentation
        all = {}
        for k in split_data.keys():
            temp = {}
            for v in split_data[k]:
                action = v.split('/')[-1]
                if action not in temp.keys():
                    temp[action] = 1
                else:
                    temp[action] += 1
            all[k] = temp
        split_data['status'] = all
        io.dumps_json(split_data, os.path.join(_DatasetPath, "split" + str(split_idx) + '.json'))
        self.data_dir = split_data[mode].copy()
        del split_data

    def __len__(self):
        if self.data_type == 'all':
            return len(self.all_data)
        else:
            return len(self.data_dir)

    def __getitem__(self, index):
        if self.task == 'recog_only':
            # load data: [len, sample_num, H, W, C]
            clips, labels, pad_num = self._load_clips_and_labels(self.data_dir[index], load_mode=self.load_mode)
            # TODO: Add preprocess here
            clips = self._normalize(clips)
            # [len, sample_num, H, W, C] --> [len, C, sample_num, H, W]
            clips = clips.transpose((0, 4, 1, 2, 3))
            
            return torch.from_numpy(clips), torch.from_numpy(labels), pad_num
        else:
            if self.data_type == 'all':
                obs_clips, obs_labels, obs_pad_num, anti_clips, anti_labels, anti_pad_num, data_dir, obs_itval, anti_itval \
                                            = self.all_data[index]
            else:
                obs_clips, obs_labels, obs_pad_num, anti_clips, anti_labels, anti_pad_num \
                                            = self._recog_anti_data_gen(self.data_dir[index])
                data_dir = self.data_dir[index]
            if self.feat_type == 'online':
                obs_clips = self._normalize(obs_clips).transpose((0, 1, 5, 2, 3, 4))
            if self.anti_feat and self.feat_type=="online":
                anti_clips = self._normalize(anti_clips).transpose((0, 1, 5, 2, 3, 4))
            # add noise
            if list(BF_CONFIG['data_noise'].keys())[0] and self.data_aug and np.random.rand() >= 0.5:
                noise_type = list(BF_CONFIG['data_noise'].keys())[0]
                np.random.seed()
                if noise_type == 'uniform':
                    noise = np.random.uniform(BF_CONFIG['data_noise'][noise_type][0], BF_CONFIG['data_noise'][noise_type][1], obs_clips.shape) 
                elif noise_type == 'normal':
                    noise = np.random.normal(BF_CONFIG['data_noise'][noise_type][0], BF_CONFIG['data_noise'][noise_type][1], obs_clips.shape)
                obs_clips = np.maximum((obs_clips + noise).astype(np.float32), 0)
            # import ipdb; ipdb.set_trace()
            return torch.from_numpy(obs_clips), torch.from_numpy(obs_labels), obs_pad_num, \
                   torch.from_numpy(anti_clips), torch.from_numpy(anti_labels), anti_pad_num, \
                   data_dir, torch.from_numpy(obs_itval), torch.from_numpy(anti_itval)

    def _recog_anti_all_data_gen(self, data_dir_list):
        all_data = []
        for data_dir in data_dir_list:
            if self.feat_type == 'online':
                frames = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir)])
                all_buffer, all_label = self._sample(data_dir, frames, 1)
            else:
                all_buffer, all_label = self.data_feat[data_dir]["avg_feat"], self.data_feat[data_dir]["label"]
                # import ipdb; ipdb.set_trace()
                all_itval = np.zeros_like(all_label, dtype=np.float32)
                l_idx, r_idx = 0, 0
                while(r_idx<=all_label.shape[0]):
                    if r_idx == all_label.shape[0]:
                        itval = r_idx - l_idx
                        all_itval[l_idx:r_idx] = itval
                        break
                    if all_label[r_idx] == all_label[l_idx]:
                        r_idx += 1
                    else:
                        itval = r_idx - l_idx
                        all_itval[l_idx:r_idx] = itval
                        l_idx = r_idx
                        r_idx += 1

            obs_perc = BF_CONFIG['train_obs_perc']
            obs_buffer_list = []; obs_label_list = []; obs_pad_num_list = []
            anti_buffer_list = []; anti_label_list = []; anti_pad_num_list = []
            data_dir_list = []; obs_itval_list = []; anti_itval_list = []
            for i in obs_perc:
                obs_content = all_buffer[:int(i*all_buffer.shape[0])]
                obs_label = all_label[:int(i*all_label.shape[0])]
                obs_itval = all_itval[:int(i*all_itval.shape[0])]
                anti_content = np.array([], np.float32)
                if self.anti_feat:
                    anti_content = all_buffer[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
                anti_label = all_label[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
                anti_itval = all_itval[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
                obs_pad_num = 0; anti_pad_num = 0
                if obs_content.shape[0] < self.video_len:
                    obs_pad_num = self.video_len - obs_content.shape[0]
                    if self.feat_type == 'online':
                        obs_content = np.concatenate((obs_content, np.tile(obs_content[-1][None,:], (obs_pad_num, 1, 1, 1, 1))))
                    else:
                        obs_content = np.concatenate((obs_content, np.tile(obs_content[-1][None,:], (obs_pad_num, 1))))
                    obs_label = np.concatenate((obs_label, np.array([-100]*obs_pad_num)))
                elif obs_content.shape[0] > self.video_len:
                    omit_idxs = sorted(random.sample((range(obs_content.shape[0])), int(obs_content.shape[0]-self.video_len)))
                    obs_content = np.delete(obs_content, omit_idxs, axis=0)
                    obs_label = np.delete(obs_label, omit_idxs)
                    obs_itval = np.delete(obs_itval, omit_idxs)
                else:
                    pass
                if anti_label.shape[0] < self.video_len:
                    anti_pad_num = self.video_len - anti_label.shape[0]
                    if self.anti_feat:
                        if self.feat_type == 'online':
                            anti_content = np.concatenate((anti_content, np.tile(anti_content[-1][None, :], (anti_pad_num, 1, 1, 1, 1))))
                        else:
                            anti_content = np.concatenate((anti_content, np.tile(anti_content[-1][None, :], (anti_pad_num, 1))))
                    anti_label = np.concatenate((anti_label, np.array([-100]*anti_pad_num)))
                elif anti_label.shape[0] > self.video_len:
                    omit_idxs = sorted(random.sample((range(anti_label.shape[0])), int(anti_label.shape[0]-self.video_len)))
                    if self.anti_feat:
                        anti_content = np.delete(anti_content, omit_idxs, axis=0)
                    anti_label = np.delete(anti_label, omit_idxs)
                    anti_itval = np.delete(anti_itval, omit_idxs)
                else:
                    pass
                obs_buffer_list.append(obs_content[None, :]); obs_label_list.append(obs_label[None,:]); obs_pad_num_list.append([obs_pad_num])
                anti_buffer_list.append(anti_content[None, :]); anti_label_list.append(anti_label[None, :]); anti_pad_num_list.append([anti_pad_num])
                data_dir_list.append(data_dir); obs_itval_list.append(obs_itval); anti_itval_list.append(anti_itval)

            for data in zip(obs_buffer_list, obs_label_list, obs_pad_num_list, anti_buffer_list, anti_label_list, anti_pad_num_list, data_dir_list, obs_itval_list, anti_itval_list):
                all_data.append(data)
        
        return all_data 

    def _recog_anti_data_gen(self, data_dir):
        if self.feat_type == 'online':
            frames = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir)])
            all_buffer, all_label = self._sample(data_dir, frames, 1)
        else:
            all_buffer, all_label = self.data_feat[data_dir]["avg_feat"], self.data_feat[data_dir]["label"]
        obs_perc = BF_CONFIG['train_obs_perc']
        obs_buffer_list = []; obs_label_list = []; obs_pad_num_list = []
        anti_buffer_list = []; anti_label_list = []; anti_pad_num_list = []
        for i in obs_perc:
            obs_content = all_buffer[:int(i*all_buffer.shape[0])]
            obs_label = all_label[:int(i*all_label.shape[0])]
            anti_content = np.array([], np.float32)
            if self.anti_feat:
                anti_content = all_buffer[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
            anti_label = all_label[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
            obs_pad_num = 0; anti_pad_num = 0
            if obs_content.shape[0] < self.video_len:
                obs_pad_num = self.video_len - obs_content.shape[0]
                if self.feat_type == 'online':
                    obs_content = np.concatenate((obs_content, np.tile(obs_content[-1][None,:], (obs_pad_num, 1, 1, 1, 1))))
                else:
                    obs_content = np.concatenate((obs_content, np.tile(obs_content[-1][None,:], (obs_pad_num, 1))))
                obs_label = np.concatenate((obs_label, np.array([-100]*obs_pad_num)))
            elif obs_content.shape[0] > self.video_len:
                omit_idxs = sorted(random.sample((range(obs_content.shape[0])), int(obs_content.shape[0]-self.video_len)))
                obs_content = np.delete(obs_content, omit_idxs, axis=0)
                obs_label = np.delete(obs_label, omit_idxs)
            else:
                pass
            if anti_label.shape[0] < self.video_len:
                anti_pad_num = self.video_len - anti_label.shape[0]
                if self.anti_feat:
                    if self.feat_type == 'online':
                        anti_content = np.concatenate((anti_content, np.tile(anti_content[-1][None, :], (anti_pad_num, 1, 1, 1, 1))))
                    else:
                        anti_content = np.concatenate((anti_content, np.tile(anti_content[-1][None, :], (anti_pad_num, 1))))
                anti_label = np.concatenate((anti_label, np.array([-100]*anti_pad_num)))
            elif anti_label.shape[0] > self.video_len:
                omit_idxs = sorted(random.sample((range(anti_label.shape[0])), int(anti_label.shape[0]-self.video_len)))
                if self.anti_feat:
                    anti_content = np.delete(anti_content, omit_idxs, axis=0)
                anti_label = np.delete(anti_label, omit_idxs)
            else:
                pass
            obs_buffer_list.append(obs_content[None, :]); obs_label_list.append(obs_label[None,:]); obs_pad_num_list.append(obs_pad_num)
            anti_buffer_list.append(anti_content[None, :]); anti_label_list.append(anti_label[None, :]); anti_pad_num_list.append(anti_pad_num)

        obs_buffer = np.concatenate(obs_buffer_list)
        obs_label = np.concatenate(obs_label_list)
        obs_pad_num = np.array(obs_pad_num_list)
        anti_buffer = np.concatenate(anti_buffer_list)
        anti_label = np.concatenate(anti_label_list)
        anti_pad_num = np.array(anti_pad_num_list)
        
        return obs_buffer, obs_label, obs_pad_num, anti_buffer, anti_label, anti_pad_num 

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
        for k, v in sorted(notation_info[data_dir].items(), key=lambda a: int(a[0].split('-')[0])):
            # temp_label = np.zeros(len(BF_ACTION_CLASS), np.dtype('float32'))
            a, b= k.split('-')
            # skip the extremely short segments
            if a==b or (int(b)-int(a)<8):
                continue
            # clip_num = (int(b)-int(a))*sample_ratio // 15 if (int(b)-int(a))*sample_ratio // 15 else 1
            clip_num = round((int(b)-int(a))*sample_ratio / 15)
            if clip_num == 1:
                if int(b) > len(frames) and len(frames)-8 < int(a):
                    continue
                temp_data = np.empty((BF_CONFIG['sample_num_each_clip'], BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
                if not (int(b) > len(frames)):
                    s = int(a) if int(b)-8==int(a) else random.choice(range(int(a), int(b)-8))
                else:
                    s = int(a) if len(frames)-8==int(a) else random.choice(range(int(a), len(frames)-8))
                for i, j in enumerate(range(s, s+8)):
                    temp_frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                    temp_data[i] = temp_frame   
                buffer_list.append(temp_data)
                # consider "walk_in" and "walk_out" as "SIL"
                if v=="walk_in" or v=="walk_out":
                    v='SIL'
                labels_list.append(BF_ACTION_CLASS.index(v))
                continue      
            else:
                if clip_num == 2:
                    s_list = [int(a), int(b)-15]
                else:
                    # random choice is redundant when save the train data offline
                    s_list = random.sample((range(int(a)+15, int(b)-15, 15)), int(clip_num-2))
                    s_list.extend([int(a), int(b)-15])
                for s in sorted(s_list):
                    temp_data = np.empty((BF_CONFIG['sample_num_each_clip'], BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32')) 
                    # For some reasons, the number of video frame is less than the max length of video in notation. 
                    if s+15 > len(frames):
                        for i, j in enumerate(range(len(frames)-8, len(frames))):
                            temp_frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                            temp_data[i] = temp_frame
                    else:
                        for i, j in enumerate(range(s, s+15, 2)):
                            temp_frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                            temp_data[i] = temp_frame
                    buffer_list.append(temp_data)
                    # consider "walk_in" and "walk_out" as "SIL"
                    if v=="walk_in" or v=="walk_out":
                        v='SIL'
                    labels_list.append(BF_ACTION_CLASS.index(v))

        buffer = np.array(buffer_list)
        labels = np.array(labels_list)
        return buffer, labels

    def _load_clips_all(self, data_dir, frames, random_sample=False):
        ori_video_len = len(frames) // BF_CONFIG['FPS']
        labels = np.zeros(ori_video_len, np.dtype('float32'))
        buffer = np.empty((ori_video_len, self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
        temp = np.empty((self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
        # TODO: optimize this part
        gt_labels_list = []
        for k, v in sorted(self.notation_info[data_dir].items(), key=lambda a: int(a[0].split('-')[0])):
            if v=="walk_in" or v=="walk_out":
                v='SIL'
            gt_labels_list.extend([v]*int(int(k.split('-')[1])-int(k.split('-')[0])+1))
        for i in range(ori_video_len):
            if random_sample:
                for idx, j in enumerate(sorted(random.sample(range(i*15, (i+1)*15), 8))):
                     # read image and convert it from BGR to RGB
                    frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                    temp[idx] = frame                
            else:
                for idx, j in enumerate(range(i*15, (i+1)*15, 2)):
                    frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
                    temp[idx] = frame
            # get the label for each clip
            labels[i] = BF_ACTION_CLASS.index(gt_labels_list[i+8]) if gt_labels_list[i+7] == gt_labels_list[i+8] else BF_ACTION_CLASS.index(gt_labels_list[i+7])
            buffer[i] = temp
        return buffer, labels

    def _normalize(self, buffer):
        try:
            # means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            # stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # # normalization
            # buffer /= 255.0
            # # buffer = (buffer - means[None, None, None, None, :]) / stds[None, None, None, None, :]
            # buffer = (buffer - means) / stds
            buffer = buffer / 255.0 * 2.0 - 1.0              # normalized to [-1, 1], which is the normalization method used in original I3D model
            return buffer
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)


class BreakfastDataset_Evaluation(BreakfastDataset):
    def __init__(self, mode='test', split_idx=0, feat_type='online', gen_feat=False, preproc=None, evaluation=True):
        super(BreakfastDataset_Evaluation, self).__init__(mode=mode, split_idx=split_idx, feat_type=feat_type, preproc=None, evaluation=evaluation)
        if not gen_feat:
            self.data = h5py.File(os.path.join(BF_CONFIG["data_dir"], f"i3d_feat_eval_split_{split_idx}_{mode}.hdf5"), 'r')
        
        self.gen_feat = gen_feat

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, index):
        frame_path = self.data_dir[index]
        if self.gen_feat:
            frames = sorted([os.path.join(frame_path, img) for img in os.listdir(frame_path)])
            all_buffer, all_label = self._load_clips_all(frame_path, frames)
            obs_buffer_list = []; obs_pad_num_list = []; anti_pad_num_list = []
            for i in BF_CONFIG['eval_obs_perc']:
                obs_pad_num = 0; anti_pad_num = 0
                obs_content = all_buffer[:int(i*all_buffer.shape[0])]
                anti_content = all_buffer[int(i*all_buffer.shape[0]):int((0.5+i)*all_buffer.shape[0])]
                if obs_content.shape[0] < self.video_len:
                    obs_pad_num = self.video_len - obs_content.shape[0]
                    obs_content = np.concatenate((obs_content, np.tile(obs_content[-1][None,:], (obs_pad_num, 1, 1, 1, 1))))
                elif obs_content.shape[0] > self.video_len:
                    n = obs_content.shape[0] - self.video_len
                    obs_content = obs_content[n:]
                else:
                    pass
                if anti_content.shape[0] < self.video_len:
                    anti_pad_num = self.video_len - anti_content.shape[0]
                else:
                    pass
                obs_buffer_list.append(obs_content[None, :]);  obs_pad_num_list.append(obs_pad_num); anti_pad_num_list.append(anti_pad_num)    

            obs_buffer = np.concatenate(obs_buffer_list)
            obs_pad_num = np.array(obs_pad_num_list)
            anti_pad_num = np.array(anti_pad_num_list)

            obs_buffer = self._normalize(obs_buffer).transpose((0, 1, 5, 2, 3, 4))

            return torch.from_numpy(obs_buffer), obs_pad_num, anti_pad_num, frame_path      
        
        else:
            return torch.from_numpy(self.data[frame_path]['feat'][:]), self.data[frame_path]['obs_pad_num'][:],\
                   self.data[frame_path]['anti_pad_num'][:], frame_path

## just for recog_anti model
def collate_fn_with_backbone(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_obs_clips = torch.empty((0, BF_CONFIG['video_len'], 3, BF_CONFIG['sample_num_each_clip'], BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH']), dtype=torch.float)
    batch_obs_labels = torch.empty((0, BF_CONFIG['video_len']), dtype=torch.int64)
    batch_obs_pad_num = np.empty((0))
    batch_anti_clips = torch.empty((0, BF_CONFIG['video_len'], 3, BF_CONFIG['sample_num_each_clip'], BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH']), dtype=torch.float)
    batch_anti_labels = torch.empty((0, BF_CONFIG['video_len']), dtype=torch.int64)
    batch_anti_pad_num = np.empty((0))
    batch_img_dir = []

    for data in batch:
        batch_obs_clips = torch.cat((batch_obs_clips, data[0]))
        batch_obs_labels = torch.cat((batch_obs_labels, data[1]))
        batch_obs_pad_num = np.concatenate((batch_obs_pad_num, data[2]))
        if len(data[3].shape) > 3:
            batch_anti_clips = torch.cat((batch_anti_clips, data[3]))
        batch_anti_labels = torch.cat((batch_anti_labels, data[4]))
        batch_anti_pad_num = np.concatenate((batch_anti_pad_num, data[5]))
        batch_img_dir.append(data[6])

    return batch_obs_clips, batch_obs_labels, batch_obs_pad_num, \
           batch_anti_clips, batch_anti_labels, batch_anti_pad_num, batch_img_dir

## just for recog_anti model
def collate_fn_without_backbone(batch):
    '''
        Default collate_fn(): https://github.com/pytorch/pytorch/blob/1d53d0756668ce641e4f109200d9c65b003d05fa/torch/utils/data/_utils/collate.py#L43
    '''
    batch_obs_clips = torch.empty((0, BF_CONFIG['video_len'], 1024), dtype=torch.float)
    batch_obs_labels = torch.empty((0, BF_CONFIG['video_len']), dtype=torch.int64)
    batch_obs_pad_num = np.empty((0))
    batch_anti_clips = torch.empty((0, BF_CONFIG['video_len'], 1024), dtype=torch.float)
    batch_anti_labels = torch.empty((0, BF_CONFIG['video_len']), dtype=torch.int64)
    batch_anti_pad_num = np.empty((0))
    batch_img_dir = []
    batch_obs_itval_gt = torch.empty((0))
    batch_anti_itval_gt = torch.empty((0))

    for data in batch:
        batch_obs_clips = torch.cat((batch_obs_clips, data[0]))
        batch_obs_labels = torch.cat((batch_obs_labels, data[1]))
        batch_obs_pad_num = np.concatenate((batch_obs_pad_num, data[2]))
        if len(data[3].shape) > 2:
            batch_anti_clips = torch.cat((batch_anti_clips, data[3]))
        batch_anti_labels = torch.cat((batch_anti_labels, data[4]))
        batch_anti_pad_num = np.concatenate((batch_anti_pad_num, data[5]))
        batch_img_dir.append(data[6])
        batch_obs_itval_gt = torch.cat((batch_obs_itval_gt, data[7]))
        batch_anti_itval_gt = torch.cat((batch_anti_itval_gt, data[8]))

    return batch_obs_clips, batch_obs_labels, batch_obs_pad_num, \
           batch_anti_clips, batch_anti_labels, batch_anti_pad_num, \
           batch_img_dir, batch_obs_itval_gt, batch_anti_itval_gt

if __name__ == "__main__":
    dataset = BreakfastDataset()
    for data in iter(dataset):
        print(data)
    print("test finished!!!")




    # def _load_clips_all(self, data_dir, frames, random_sample=False):
    #     ori_video_len = len(frames) // BF_CONFIG['FPS']
    #     labels = np.zeros((ori_video_len, len(BF_ACTION_CLASS)), np.dtype('float32'))
    #     buffer = np.empty((ori_video_len, self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
    #     temp = np.empty((self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
    #     # TODO: optimize this part
    #     for i in range(ori_video_len):
    #         f4_idx = 0; f5_idx = 0
    #         if random_sample:
    #             for idx, j in enumerate(sorted(random.sample(range(i*15, (i+1)*15), 8))):
    #                  # read image and convert it from BGR to RGB
    #                 frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
    #                 temp[idx] = frame        
    #                 if idx == 3: f4_idx = j 
    #                 if idx == 4: f5_idx = j           
    #         else:
    #             for idx, j in enumerate(range(i*15, (i+1)*15, 2)):
    #                 frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
    #                 temp[idx] = frame
    #                 if idx == 3: f4_idx = j 
    #                 if idx == 4: f5_idx = j 
    #         # get the label for each clip
    #         f4_label = None; f5_label = None
    #         for k, v in self.notation_info[data_dir].items():
    #             if f4_label and f5_label:
    #                 break
    #             a, b = k.split('-')
    #             if f4_idx+1 in range(int(a), int(b)):
    #                 f4_label = v
    #             if f5_idx+1 in range(int(a), int(b)):
    #                 f5_label = v
    #         labels[i][BF_ACTION_CLASS.index(f5_label if f4_label == f5_label else f4_label)] = 1
    #         buffer[i] = temp
    #     return buffer, labels


    # def _sample(self, data_dir, frames, sample_ratio=1):
    #     labels_list = []
    #     buffer_list = []
    #     # sample the frames within each segment
    #     for k, v in sorted(self.notation_info[data_dir].items(), key=lambda a: int(a[0].split('-')[0])):
    #         # temp_label = np.zeros(len(BF_ACTION_CLASS), np.dtype('float32'))
    #         a, b= k.split('-')
    #         # skip the extremely short segments
    #         if a==b or (int(b)-int(a)<8):
    #             continue
    #         # clip_num = (int(b)-int(a))*sample_ratio // 15 if (int(b)-int(a))*sample_ratio // 15 else 1
    #         clip_num = round((int(b)-int(a))*sample_ratio / 15)
    #         try:
    #             for s in sorted(random.sample((range(int(a), int(b), 15)), int(clip_num))):
    #                 temp_data = np.empty((self.sample_num, BF_CONFIG['RESIZE_HEIGHT'], BF_CONFIG['RESIZE_WIDTH'], 3), np.dtype('float32'))
    #                 # to ensure each segment have at least one clip
    #                 if clip_num == 1:
    #                     if int(b) > len(frames):
    #                         break
    #                     s = int(a) if int(b)-8==int(a) else random.choice(range(int(a), int(b)-8))
    #                     for i, j in enumerate(range(s, s+8)):
    #                         frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
    #                         temp_data[i] = frame   
    #                     # temp_label[BF_ACTION_CLASS.index(v)] = 1 
    #                     # labels_list.append(temp_label)
    #                     buffer_list.append(temp_data)
    #                     if v=="walk_in" or v=="walk_out":
    #                         v='SIL'
    #                     labels_list.append(BF_ACTION_CLASS.index(v))
    #                     break   
    #                 # set "flag" to avoid the empty temp_data                 
    #                 flag = False
    #                 if s+15 <= int(b):
    #                     if s+15 > len(frames):
    #                         break
    #                     flag = True
    #                     for i, j in enumerate(range(s, s+15, 2)):
    #                         frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
    #                         temp_data[i] = frame
    #                     # temp_label[BF_ACTION_CLASS.index(v)] = 1
    #                     if v=="walk_in" or v=="walk_out":
    #                         v='SIL'
    #                     temp_label = BF_ACTION_CLASS.index(v) 
    #                 elif s+8 <= int(b):
    #                     if s+8 > len(frames):
    #                         break
    #                     flag = True
    #                     for i, j in enumerate(range(int(s), int(s)+8)):
    #                         frame = cv2.imread(frames[j]).astype(np.float32)[:,:,::-1]
    #                         temp_data[i] = frame
    #                     # temp_label[BF_ACTION_CLASS.index(v)] = 1
    #                     if v=="walk_in" or v=="walk_out":
    #                         v='SIL'
    #                     temp_label = BF_ACTION_CLASS.index(v)
    #                 if flag:
    #                     buffer_list.append(temp_data)
    #                     labels_list.append(temp_label)
    #         except Exception as e:
    #             import ipdb; ipdb.set_trace()
    #             print(e)
    #     buffer = np.array(buffer_list)
    #     labels = np.array(labels_list)
    #     return buffer, labels
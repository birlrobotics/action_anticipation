# To avoid error if ROS is installed in the device
import sys
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)

import torchvision.models._utils as _utils
from model.i3d import I3D
from dataset.config import BF_CONFIG, BF_ACTION_CLASS
import utils.io as io
import h5py
import os
import torch
import numpy as np
import random
import cv2
import tqdm

def make_backbone(name='i3d', ck_dir=None, fixed=True):
    model = I3D()
    checkpoints = torch.load(ck_dir)
    model.load_state_dict(checkpoints)
    if fixed:
        for param in model.parameters():
            param.requires_grad = False
    return model.features

def normalize(buffer):    
    # means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # # normalization
    # buffer /= 255.0
    # # buffer = (buffer - means[None, None, None, None, :]) / stds[None, None, None, None, :]
    # buffer = (buffer - means) / stds
    buffer = buffer / 255.0 * 2.0 - 1.0              # normalized to [-1, 1], which is the normalization method used in original I3D model
    return buffer

def sample(data_dir, frames, sample_ratio=1):
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

if __name__ == "__main__":
    backbone = make_backbone(name=BF_CONFIG["backbone"], ck_dir=BF_CONFIG["cp_dir"], fixed=BF_CONFIG["fixed"])
    backbone = _utils.IntermediateLayerGetter(backbone, BF_CONFIG['RETURN_LAYERS'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone.to(device)
    backbone.eval()

    feat_file_name = h5py.File(os.path.join(BF_CONFIG["data_dir"], BF_CONFIG["feat_hdf5_name"]), 'w')
    notation_info = io.loads_json(os.path.join(BF_CONFIG["data_dir"], "notation.json"))
    for data_dir in tqdm.tqdm(list(notation_info.keys())):
        # import ipdb; ipdb.set_trace()
        # data_dir = './dataset/breakfast/rgb_frame/P51/webcam02/cereals'
        frames = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir)])
        all_buffer, all_label = sample(data_dir, frames, 1)
        all_buffer = normalize(all_buffer).transpose((0, 4, 1, 2, 3))
        all_buffer = torch.from_numpy(all_buffer).to(device)
        if (all_buffer.shape[0] >= 500):
            temp_feat = backbone(all_buffer[:500])
            feat = temp_feat['feat']
            temp_feat = backbone(all_buffer[500:])
            feat = torch.cat((feat, temp_feat['feat']))
            feat = feat.squeeze()
        else:
            feat = backbone(all_buffer)
            feat = feat['feat'].squeeze()

        feat_file_name.create_group(data_dir)
        feat_file_name[data_dir].create_dataset(name='avg_feat', data=feat.cpu().detach().numpy())
        feat_file_name[data_dir].create_dataset(name='label', data=all_label)
        # torch.cuda.empty_cache()
    
    feat_file_name.close()
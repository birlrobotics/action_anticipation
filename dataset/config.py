import utils.io as io

# Breakfast
BF_CONFIG = {
    # For data preparation
    "data_dir": "./dataset/breakfast",
    "test_size": 0.2,                 # For splitting the trainval setting into train ans val
    "FPS": 15,
    "EXTRACT_FREQUENCY": 1,
    "RESIZE_WIDTH": 224,
    "RESIZE_HEIGHT": 224,
    "RETURN_LAYERS": {'16': 'feat'},  # {'15': 'Mixed_5c', '16': 'AvgPool'}
    "load_mode": "uniform",           # [all, uniform], how to load the clips
    "video_len": 300,                 # uniformly sample clips, only available when the sample_mode is "uniform"
    "sample_num_each_clip": 8,
    "feat_hdf5_name": "i3d_feat_48class.hdf5",  # "i3d_feat_50class.hdf5" take "walk_in" and "walk_out" into account
    "obs_perc": [.2, .3, .5], 
    # For transformer
    "n_layers": 2,
    "n_attn_head": 8,
    "d_input": 1024,
    "d_inner": 2048,
    "d_qk": 64,
    "d_v": 64,
    "drop_prob": 0.1,
    "pos_enc": True,
    "queries_norm_factor": 60.,
    "return_attn": True,
    # For backbone
    "backbone": 'i3d',
    "cp_dir": "./checkpoints/i3d/rgb_imagenet.pkl",
    "fixed": True,
    # For training
    "recog_weight": 1,
    "anti_weight": 1, 
}

# 50 actions in total, but action "walk in" and "walk out" are not included in the original paper (48 fine-grained actions)
BF_ACTION_CLASS = [i.rstrip() for i in io.read('./dataset/breakfast/action_class.txt', read_type='readlines')]
# consider "walk_in" and "walk_out" as SIL
BF_ACTION_CLASS.remove("walk_out")
BF_ACTION_CLASS.remove("walk_in")

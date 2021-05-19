import os
import h5py
import argparse

import torch
from model.main import Anticipation_Without_Backbone
from dataset.config import BF_CONFIG, BF_ACTION_CLASS
import utils.io as io
from utils.vis_tool import VISUALIZER

def arg_parse():
    parser = argparse.ArgumentParser(description='visualizing the results')
    parser.add_argument('--ck', type=str, default='result/breakfast_checkpoint.pth',
                        help='path to the pretrained checkpoints')
    parser.add_argument('--file', type=str, default='./dataset/breakfast/rgb_frame/P25/webcam02/friedegg',
                        help='path to the video you want to visualize, default=./dataset/breakfast/rgb_frame/P03/cam01/cereals')
    parser.add_argument('--mode', type=str, default='test',
                        help='training sample or testing sample or val sample, default=test')
    parser.add_argument('--obs_prec', type=float, nargs='+', default=[.2, .3, .5],
                        help='Which precentage you want to use. To use: --obs_prec .2 .3 ... , default=[.2, .3, .5].')
    parser.add_argument('--save_dir', type=str, default='./result/breakfast',
                        help='path to save the visualization results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # prepare the data
    default_obs_prec_dict, obs_prec_dict = {}, {}
    for i, j in enumerate(BF_CONFIG['eval_obs_perc']):
        default_obs_prec_dict[str(j)] = i
    for i in args.obs_prec:
        obs_prec_dict[str(i)] = default_obs_prec_dict[str(i)]
    gt_info = io.loads_json(os.path.join(BF_CONFIG['data_dir'], "notation.json"))
    gt_labels_list = []
    for k, v in sorted(gt_info[args.file].items(), key=lambda a: int(a[0].split('-')[0])):
        if v=="walk_in" or v=="walk_out":
            v='SIL'
        gt_labels_list.extend([v]*int(int(k.split('-')[1])-int(k.split('-')[0])+1))

    all_data = h5py.File(os.path.join(BF_CONFIG["data_dir"], f"i3d_feat_eval_split_0_{args.mode}.hdf5"), 'r')
    obs_feat = torch.from_numpy(all_data[args.file]['feat'][:]).to(device)
    obs_pad_num = all_data[args.file]['obs_pad_num'][:]
    anti_pad_num = all_data[args.file]['anti_pad_num'][:]

    # construct the model
    model = Anticipation_Without_Backbone(train=False)
    checkpoint = torch.load(args.ck)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    # infence
    save_dir = os.path.join(args.save_dir, 'video', args.file.split('/')[-3]+'_'+args.file.split('/')[-2]+'_'+args.file.split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis = VISUALIZER(v_dir=save_dir)
    with torch.no_grad():
        recog_logits, anti_logits, *attn = model(obs_feat, obs_pad_num, anti_pad_num)
        recog_scores, anti_scores = torch.nn.Softmax(-1)(recog_logits), torch.nn.Softmax(-1)(anti_logits)
        top_recog_probs, top_recog_class = recog_scores.topk(1, dim=-1)
        top_anti_probs, top_anti_class = anti_scores.topk(1, dim=-1)
        for k, v in obs_prec_dict.items():
            recog_res = top_recog_class[v][:int(obs_feat.shape[1]-obs_pad_num[v])].squeeze().cpu()
            anti_res = top_anti_class[v][:int(obs_feat.shape[1]-anti_pad_num[v])].squeeze().cpu()
            vis.show(args.file, gt_labels_list, recog_res, anti_res, k)


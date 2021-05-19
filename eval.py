''' Evaluation script for action recognition and anticipation '''
''' Adatped from project: https://github.com/yabufarha/anticipating-activities'''

import os
import copy
import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset.config import BF_CONFIG, BF_ACTION_CLASS
from model.model import Anticipation_Without_Backbone, Anticipation_With_Backbone
from dataset.breakfast_dataset import BreakfastDataset_Evaluation
import utils.io as io

import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Anticipation Training.")
    # For model
    parser.add_argument('--ck', type=str, default=None, 
                        help='the path to the specified checkpoint')
    # For dataset
    parser.add_argument('--split_idx', type=int, default=0, choices=[0,1,2,3], 
                        help='dataset splited configuration: default=0')
    parser.add_argument('--ds', '--dataset', type=str, default='breakfast', 
                        help='The dataset you want to train: default=breakfast')
    parser.add_argument('--ow_feat', action='store_true',  
                        help='overwrite the backbone  feature')
    parser.add_argument('--mode', type=str, default='test', 
                        help='default=test')    
    return parser.parse_args()

def evaluation():
    # create the directory to the result file
    ck_ver, exp_ver, ds_name = args.ck.split('/')[-1], args.ck.split('/')[-2], args.ck.split('/')[-3]
    save_dir = os.path.join('./result', ds_name, exp_ver, str(args.split_idx), ck_ver, args.mode)
    io.mkdir_if_not_exists(save_dir, recursive=True)

    # prepare data
    if args.ds == 'breakfast':
        test_set = BreakfastDataset_Evaluation(mode=args.mode, split_idx=args.split_idx, gen_feat=False, evaluation=True)
        test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
        print(F"Finish preparing testing data for {args.ds} split {args.split_idx}.")

    # load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ck, map_location=device)

    # set up the model. NOTE: you need to update the 'dataset/config.py' file based on the saving configuration of the checkpoints
    model = Anticipation_Without_Backbone(train=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Contructed model. Loaded checkpoint. Testing on {}.".format(device))

    # Define some variable for saving the results
    n_T_anti = {}
    n_T_recog = {}
    for i in BF_CONFIG['eval_obs_perc']:
        temp = {}
        for j in BF_CONFIG['pred_perc']:
            temp[str(j)] = np.zeros(len(BF_ACTION_CLASS))
        n_T_anti[str(i)] = temp
    for i in BF_CONFIG['eval_obs_perc']:
        n_T_recog[str(i)] = np.zeros(len(BF_ACTION_CLASS))
    n_F_anti = copy.deepcopy(n_T_anti)
    n_F_recog = copy.deepcopy(n_T_recog)
    raw_res = {} 

    # For evaluation
    for data in tqdm(test_dataloader):
        obs_feat = data[0][0].to(device)
        obs_pad_num = data[1][0]
        anti_pad_num = data[2][0]
        data_dir = data[3][0]
        
        raw_res[data_dir] = {}
        gt_labels_list = []
        for k, v in sorted(gt_info[data_dir].items(), key=lambda a: int(a[0].split('-')[0])):
            if v=="walk_in" or v=="walk_out":
                v='SIL'
            gt_labels_list.extend([BF_ACTION_CLASS.index(v)]*int(int(k.split('-')[1])-int(k.split('-')[0])+1))
        
        with torch.no_grad():
            recog_logits, anti_logits, *attn = model(obs_feat, obs_pad_num, anti_pad_num)
            # import ipdb; ipdb.set_trace()
            recog_scores, anti_scores = torch.nn.Softmax(-1)(recog_logits), torch.nn.Softmax(-1)(anti_logits) 
            top_recog_probs, top_recog_class = recog_scores.topk(1, dim=-1)
            top_anti_probs, top_anti_class = anti_scores.topk(1, dim=-1)
            for i, j in enumerate(BF_CONFIG['eval_obs_perc']):
                # For recognition
                recog_res = top_recog_class[i][:int(obs_feat.shape[1]-obs_pad_num[i])].squeeze().cpu()
                for k in range(recog_res.shape[0]):
                    for z in range(k*15, (k+1)*15):
                        if gt_labels_list[z] == recog_res[k]:
                            n_T_recog[str(j)][gt_labels_list[z]] += 1
                        else:
                            n_F_recog[str(j)][gt_labels_list[z]] += 1
                # For anticipation
                s_idx = recog_res.shape[0] 
                anti_res = top_anti_class[i][:int(obs_feat.shape[1]-anti_pad_num[i])].squeeze().cpu()
                for p in BF_CONFIG['pred_perc']:
                    # the default anticipation percent is set to 50%
                    anti_len = round(p / 0.5 * anti_res.shape[0])
                    for k in range(anti_len):
                        for z in range((k+s_idx)*15, (k+s_idx+1)*15):
                            if gt_labels_list[z] == anti_res[k]:
                                n_T_anti[str(j)][str(p)][gt_labels_list[z]] += 1
                            else:
                                n_F_anti[str(j)][str(p)][gt_labels_list[z]] += 1
                raw_res[data_dir][str(j)] = {"recog_res": [BF_ACTION_CLASS[i] for i in recog_res], \
                                              "anti_res": [BF_ACTION_CLASS[i] for i in  anti_res]}
    
    # save raw results
    io.dumps_json(raw_res, os.path.join(save_dir, 'raw_result.json'))
    # save and print final results
    for p in BF_CONFIG['eval_obs_perc']:
        print('\n{:-^50}'.format(str(p)+' observation'))
        with open(os.path.join(save_dir, f'obs{str(p)}.txt'), 'w') as f:
            recog_acc_list = []
            f.write('{:-^50} \n'.format('recognition'.upper()))
            f.write('{: <20}{: <20}{: <10} \n'.format('ACTION', 'n_T / n_F', 'Acc.'))
            for i in range(len(BF_ACTION_CLASS)):
                acc = round(n_T_recog[str(p)][i] / (n_T_recog[str(p)][i]+n_F_recog[str(p)][i]) if n_T_recog[str(p)][i]+n_F_recog[str(p)][i] else 0.0, 4)
                if n_T_recog[str(p)][i] + n_F_recog[str(p)][i] !=0:
                    recog_acc_list.append(acc)
                f.write('{: <20}{: <20}{: <10} \n'.format(BF_ACTION_CLASS[i], \
                                                          str(n_T_recog[str(p)][i])+' / '+str(n_F_recog[str(p)][i]), \
                                                          acc))
            f.write('{: <20}{: <20}{: <10} \n \n'.format(' ', \
                                                         ' ', \
                                                         round(sum(recog_acc_list) / len(recog_acc_list), 4)))
            print(f'Recognition Acc:  {round(sum(recog_acc_list) / len(recog_acc_list), 4)}')

            for k in n_T_anti[str(p)].keys():
                anti_acc_list = []
                f.write('{:-^50} \n'.format(k+' anticipation'.upper()))
                f.write('{: <20}{: <20}{: <10} \n'.format('ACTION', 'n_T / n_F', 'Acc.'))
                for i in range(len(BF_ACTION_CLASS)):
                    acc = round(n_T_anti[str(p)][k][i] / (n_T_anti[str(p)][k][i]+n_F_anti[str(p)][k][i]) if (n_T_anti[str(p)][k][i]+n_F_anti[str(p)][k][i]) else 0.0, 4)
                    if n_T_anti[str(p)][k][i]+n_F_anti[str(p)][k][i] !=0:
                        anti_acc_list.append(acc)
                    f.write('{: <20}{: <20}{: <10} \n'.format(BF_ACTION_CLASS[i], \
                                                              str(n_T_anti[str(p)][k][i])+' / '+str(n_F_anti[str(p)][k][i]), \
                                                              acc))
                f.write('{: <20}{: <20}{: <10} \n \n'.format(' ', \
                                                             ' ', \
                                                             round(sum(anti_acc_list) / len(anti_acc_list), 4)))
                print(f'{k} Anticipation Acc:  {round(sum(anti_acc_list) / len(anti_acc_list), 4)}')

def create_backbone_feat():
    # prepare data
    if args.ds == 'breakfast':
        test_set = BreakfastDataset_Evaluation(mode=args.mode, split_idx=args.split_idx, gen_feat=True)
        test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
        print(F"Generate backbone feature for {args.ds} split {args.split_idx}.")
    # set up the model. NOTE: you need to update the 'dataset/config.py' file based on the saving configuration of the checkpoints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing on {}.".format(device))
    backbone = Anticipation_With_Backbone(train=False).to(device)
    backbone.eval()

    feat_file_name = h5py.File(os.path.join("dataset", args.ds, f"i3d_feat_eval_split_{args.split_idx}_{args.mode}.hdf5"), 'w')
    for data in tqdm(test_dataloader):
        obs_feat = data[0][0]
        obs_pad_num = data[1][0]
        anti_pad_num = data[2][0]
        data_dir = data[3][0]
        backbone_feat = None
        for i in range(obs_feat.shape[0]):
            feat = backbone(obs_feat[i].to(device))['feat']
            feat = feat.squeeze()
            backbone_feat = torch.cat((backbone_feat, feat[None, :])) if backbone_feat is not None else feat[None, :]
        feat_file_name.create_group(data_dir)
        feat_file_name[data_dir].create_dataset(name='feat', data=backbone_feat.cpu().detach().numpy())
        feat_file_name[data_dir].create_dataset(name='obs_pad_num', data=obs_pad_num)
        feat_file_name[data_dir].create_dataset(name='anti_pad_num', data=anti_pad_num)
    feat_file_name.close()

if __name__ == "__main__":
    args = arg_parse()
    gt_info = io.loads_json(os.path.join(BF_CONFIG['data_dir'], "notation.json"))
    if args.ow_feat or not os.path.exists(os.path.join("dataset", args.ds, f"i3d_feat_eval_split_{args.split_idx}_{args.mode}.hdf5")):
        create_backbone_feat()
    evaluation()

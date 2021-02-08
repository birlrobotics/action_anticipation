import os 
import time
from tqdm import tqdm 

import torch 
import torchvision 
from torch import nn, optim 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.config import BF_CONFIG, BF_ACTION_CLASS
from model.main import Anticipation_With_Backbone, Anticipation_Without_Backbone
from dataset.breakfast_dataset import BreakfastDataset, collate_fn_with_backbone, collate_fn_without_backbone
import utils.io as io

import argparse 

# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

def arg_parse():
    parser = argparse.ArgumentParser(description="Anticipation Training.")
    # For model
    parser.add_argument('--use_dec', action="store_false", 
                        help='use decoder or not: action="store_false')
    # For dataset
    parser.add_argument('--split_idx', type=int, default=0, choices=[0,1,2,3], 
                        help='dataset splited configuration: default=0')
    parser.add_argument('--task', type=str, default='recog_anti', choices=["recog_only", "recog_anti"],
                        help="which task do you want to conduct [recog_only or recog_anti]")
    parser.add_argument('--anti_feat', action="store_true",
                        help="return anticipation features or not: action='store_true'")
    parser.add_argument('--feat_type', type=str, default='offline', choices=["offline", "online"],
                        help="which type of feature do you want to use [offline or online]")
    # For training
    parser.add_argument('--ds', '--dataset', type=str, default='breakfast', 
                        help='The dataset you want to train: default=breakfast')
    parser.add_argument('--nw', '--num_workers', type=int, default=0, 
                        help='Number of workers used in dataloading: default=0')
    parser.add_argument('--bs', '--batch_size', type=int, default=1, 
                        help='the size of minibatch: default=1')
    parser.add_argument('--optim', type=str, default='adam', 
                        help='which optimizer to be used: default=adam')
    parser.add_argument('--lr', type=float, default=0.00001, 
                        help='learning rate: default=0.0001')
    parser.add_argument('--warmup', action="store_true", 
                        help='warmup stratery: action="store_true"')
    parser.add_argument('--epoch', type=int, default=300, 
                        help='Number of training epoch: default=100')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='number of beginning epochs : 0')
    # For logging or saving
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='path to save the log data like loss\accuracy... : ./log') 
    parser.add_argument('--exp_ver', '--e_v', type=str, default='v1', 
                        help='the version of code, will create subdir in log/ && checkpoints/ ')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='path to save the checkpoints: ./checkpoints')
    parser.add_argument('--print_every', type=int, default=10,
                        help='number of steps for printing training and validation loss: 10')
    parser.add_argument('--save_every', type=int, default=20,
                        help='number of steps for saving the model parameters: 20')
    return parser.parse_args() 


def train_model_recog_anti():
    # prepare the data
    collate_fn = collate_fn_with_backbone if args.feat_type == "online" else collate_fn_without_backbone
    train_set = BreakfastDataset(mode="train", split_idx=args.split_idx, task=args.task, feat_type=args.feat_type, anti_feat=args.anti_feat, preproc=None, over_write=False)
    val_set = BreakfastDataset(mode="val", split_idx=args.split_idx, task=args.task, feat_type=args.feat_type, anti_feat=args.anti_feat, preproc=None, over_write=False)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, collate_fn=collate_fn)
    dataset = {"train": train_set, "val": val_set}
    dataloader = {"train": train_dataloader, "val": val_dataloader}
    phase_list = ["train", 'val']
    print("Preparing data done!!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on {}.".format(device))

    # prepare the model
    assert args.use_dec == (args.task=="recog_anti"), f"args.use_dec(={args.use_dec}) should be TRUE if args.task(={args.task}) == recog_anti, vice versa." 
    if args.feat_type == "online":
        model = Anticipation_With_Backbone(use_dec=args.use_dec)
    else:
        model = Anticipation_Without_Backbone(use_dec=args.use_dec)
    model.to(device)

    # get the numbers of parameters of the designed model
    param_dict = {}
    for param in model.named_parameters():
        moduler_name = param[0].split('.')[0]
        if moduler_name in param_dict.keys():
            param_dict[moduler_name] += param[1].numel()
        else:
            param_dict[moduler_name] = param[1].numel()
    for k, v in param_dict.items():
        print(f"{k} Parameters: {v / 1e6} million.")
    print(f"Parameters in total: {sum(param_dict.values()) / 1e6} million.")

    # build optimizer && criterion
    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0)
    elif args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0, amsgrad=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    recog_criterion = nn.CrossEntropyLoss(reduction='sum')    # 'mean' 'sum' 'none'
    anti_criterion = nn.CrossEntropyLoss(reduction='sum')

    #set up logger
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.ds + '/' + args.exp_ver)
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.ds, args.exp_ver), recursive=True)
    
    # start training
    time_seq = torch.arange(1, BF_CONFIG['video_len']+1).float().to(device)[None,:] / BF_CONFIG["queries_norm_factor"]
    t1 = time.time()
    train_iter_num = 0
    for epoch in range(args.start_epoch, args.epoch):
        # warmup strategy
        if args.warmup:
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr*0.01
            if epoch == 10:                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr*0.1
            if epoch == 20:                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
        loss_list = []
        for phase in phase_list:
            s_t = time.time()
            recog_epoch_loss = 0
            anti_epoch_loss = 0
            recog_sample_num = 0
            anti_sample_num = 0
            # count = 0
            for data in tqdm(dataloader[phase]):
                # import ipdb; ipdb.set_trace()
                # count += 1; 
                # if count > 10: break
                obs_feat = data[0]
                obs_labels = data[1]
                obs_pad_num = data[2]
                anti_feat = data[3]
                anti_labels = data[4]
                anti_pad_num = data[5]
                obs_feat, obs_labels, anti_feat, anti_labels = obs_feat.to(device), obs_labels.to(device), anti_feat.to(device), anti_labels.to(device)
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    recog_logits, anti_logits, *attn = model(obs_feat, time_seq, obs_pad_num, anti_pad_num)
                    recog_loss = recog_criterion(recog_logits.reshape(recog_logits.shape[:-1].numel(), recog_logits.shape[-1]), obs_labels.reshape(obs_labels.shape.numel()))
                    anti_loss = anti_criterion(anti_logits.reshape(anti_logits.shape[:-1].numel(), anti_logits.shape[-1]), anti_labels.reshape(anti_labels.shape.numel()))
                    loss = recog_loss * BF_CONFIG["recog_weight"] + anti_loss * BF_CONFIG["anti_weight"]
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        recog_logits, anti_logits, *attn = model(obs_feat, time_seq, obs_pad_num, anti_pad_num)
                        recog_loss = recog_criterion(recog_logits.reshape(recog_logits.shape[:-1].numel(), recog_logits.shape[-1]), obs_labels.reshape(obs_labels.shape.numel()))
                        anti_loss = anti_criterion(anti_logits.reshape(anti_logits.shape[:-1].numel(), anti_logits.shape[-1]), anti_labels.reshape(anti_labels.shape.numel()))
                        loss = recog_loss * BF_CONFIG["recog_weight"] + anti_loss * BF_CONFIG["anti_weight"]
                # epoch_loss += loss.item() * (labels.shape.numel() - pad_num.sum()).float()
                recog_epoch_loss += recog_loss.item()
                anti_epoch_loss += anti_loss.item()
                recog_sample_num += (obs_labels.shape.numel() - obs_pad_num.sum())
                anti_sample_num += (anti_labels.shape.numel() - anti_pad_num.sum())
                # plot training loss iteration by tieration
                if phase == 'train':
                    r_loss = recog_loss.item()/(obs_labels.shape.numel() - obs_pad_num.sum())
                    a_loss = anti_loss.item()/(anti_labels.shape.numel() - anti_pad_num.sum())
                    i_loss = r_loss + a_loss
                    writer.add_scalars('train_iter_loss', {'recog': r_loss, 'anti': a_loss, 'all': i_loss}, train_iter_num)
                    train_iter_num += 1
            recog_epoch_loss /= recog_sample_num
            anti_epoch_loss /= anti_sample_num
            loss_list.append([recog_epoch_loss, anti_epoch_loss])
            # print loss
            if epoch == 0 or (epoch % args.print_every) == 9:
                e_t = time.time()
                print(f"Phase:[{phase}] Epoch:[{epoch+1}/{args.epoch}]  Recog_Loss:[{round(recog_epoch_loss, 4)}] Anti_Loss:[{round(anti_epoch_loss, 4)}] Execution_time:[{round(e_t-s_t, 1)}] second")

        # plot loss
        assert len(phase_list) == len(loss_list)
        if len(phase_list) == 2:
            writer.add_scalars('train_val_epoch_loss', {'train_loss': sum(loss_list[0]), 'train_recog_loss': loss_list[0][0], 'train_anti_loss': loss_list[0][1], \
                                                        'val_loss': sum(loss_list[1]), 'val_recog_loss': loss_list[1][0], 'val_anti_loss': loss_list[1][1]}, epoch)
        else:
            writer.add_scalars('trainval_epoch_loss', {'trainval_loss': sum(loss_list[0]), 'recog_loss': loss_list[0][0], 'anti_loss': loss_list[0][1]}, epoch)
        # save training information and checkpoint
        if epoch % args.save_every == (args.save_every - 1) and epoch >= 0:
            opts = {'lr': args.lr, 'b_s': args.bs, 'optim': args.optim, 'use_dec': args.use_dec}
            save_info = {"arguments": opts, "config": BF_CONFIG}
            io.dumps_json(save_info, os.path.join(args.save_dir, args.ds, args.exp_ver, 'training_info.json'))
            save_name = "checkpoint_" + str(epoch+1) + "_epoch.pth"
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.ds, args.exp_ver, save_name))

    writer.close()
    t2 = time.time()
    print("Training finished! It takes {} seconds.".format(round(t2-t1, 1)))

def train_model_recog_only():
    # prepare the data
    collate_fn = collate_fn_with_backbone if args.feat_type == "online" else collate_fn_without_backbone
    train_set = BreakfastDataset(mode="train", split_idx=args.split_idx, task=args.task, feat_type=args.feat_type, anti_feat=args.anti_feat, preproc=None, over_write=False)
    val_set = BreakfastDataset(mode="val", split_idx=args.split_idx, task=args.task, feat_type=args.feat_type, anti_feat=args.anti_feat, preproc=None, over_write=False)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, collate_fn=collate_fn)
    dataset = {"train": train_set, "val": val_set}
    dataloader = {"train": train_dataloader, "val": val_dataloader}
    phase_list = ["train", 'val']
    print("Preparing data done!!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on {}.".format(device))

    # prepare the model
    assert args.use_dec == (args.task=="recog_anti"), "args.use_dec should be TRUE if args.task==recog_anti, vice versa" 
    if args.feat_type == "online":
        model = Anticipation_With_Backbone(use_dec=args.use_dec)
    else:
        model = Anticipation_Without_Backbone(use_dec=args.use_dec)
    model.to(device)

    # get the numbers of parameters of the designed model
    param_dict = {}
    for param in model.named_parameters():
        moduler_name = param[0].split('.')[0]
        if moduler_name in param_dict.keys():
            param_dict[moduler_name] += param[1].numel()
        else:
            param_dict[moduler_name] = param[1].numel()
    for k, v in param_dict.items():
        print(f"{k} Parameters: {v / 1e6} million.")
    print(f"Parameters in total: {sum(param_dict.values()) / 1e6} million.")

    # build optimizer && criterion
    if args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0)
    elif args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0, amsgrad=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    # for param_group in model.param_groups:
        # param_group['lr'] = new_lr
    # rec_criterion = nn.BCEWithLogitsLoss()
    rec_criterion = nn.CrossEntropyLoss(reduction='sum')    # 'mean' 'sum' 'none'
    
    #set up logger
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.ds + '/' + args.exp_ver)
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.ds, args.exp_ver), recursive=True)
    
    # start training
    for epoch in range(args.start_epoch, args.epoch):
        loss_list = []
        for phase in phase_list:
            s_t = time.time()
            epoch_loss = 0
            sample_num = 0
            # count = 0
            for data in tqdm(dataloader[phase]):
                import ipdb; ipdb.set_trace()
                # count += 1; 
                # if count > 10: break
                feat = data[0]
                labels = data[1]
                pad_num = data[2]
                feat, labels = feat.to(device), labels.to(device)
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    logits = model(feat, pad_num)
                    loss = rec_criterion(logits.reshape(logits.shape[:-1].numel(), logits.shape[-1]), labels.reshape(labels.shape.numel()))
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        logits = model(feat, pad_num)
                        loss = rec_criterion(logits.reshape(logits.shape[:-1].numel(), logits.shape[-1]), labels.reshape(labels.shape.numel()))
                # epoch_loss += loss.item() * (labels.shape.numel() - pad_num.sum()).float()
                epoch_loss += loss.item()
                sample_num += (labels.shape.numel() - pad_num.sum()).float()
            epoch_loss /= sample_num
            loss_list.append(epoch_loss)
            # print loss
            if epoch == 0 or (epoch % args.print_every) == 9:
                e_t = time.time()
                print(f"Phase:[{phase}] Epoch:[{epoch+1}/{args.epoch}]  Loss:[{epoch_loss}]  Execution_time:[{round(e_t-s_t, 1)}] second")
        # plot loss
        assert len(phase_list) == len(loss_list)
        if len(phase_list) == 2:
            writer.add_scalars('train_val_loss', {'train': loss_list[0], 'val': loss_list[1]}, epoch)
        else:
            writer.add_scalars('trainval_loss', {'trainval': loss_list[0]}, epoch)
        # save training information and checkpoint
        if epoch % args.save_every == (args.save_every - 1) and epoch >= 0:
            opts = {'lr': args.lr, 'b_s': args.bs, 'optim': args.optim, 'use_dec': args.use_dec}
            save_info = {"arguments": opts, "config": BF_CONFIG, 'parameters': param_dict}
            io.dumps_json(save_info, os.path.join(args.save_dir, args.ds, args.exp_ver, 'training_info.json'))
            save_name = "checkpoint_" + str(epoch+1) + "_epoch.pth"
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.ds, args.exp_ver, save_name))

    writer.close()
    print("Training finished!!!")
        

if __name__ == "__main__":
    args = arg_parse()
    if args.task == "recog_only":
        train_model_recog_only()
    else:
        train_model_recog_anti()
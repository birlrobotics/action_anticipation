import os 
import time
from tqdm import tqdm 

import torch 
import torchvision 
from torch import nn, optim 
from torch.utils.data import DataLoader 
from tensorboardX import SummaryWriter

from dataset.config import BF_CONFIG
from model.main import Anticipation 
from dataset.breakfast_dataset import BreakfastDataset 
import utils.io as io

import argparse 

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def arg_parse():
    parser = argparse.ArgumentParser(description="Anticipation Training.")
    # For model
    parser.add_argument('--use_dec', action="store_true", 
                        help='use decoder or not: action="store_true"')
    # For training
    parser.add_argument('--ds', '--dataset', type=str, default='breakfast', 
                        help='The dataset you want to train: default=breakfast')
    parser.add_argument('--split_idx', type=int, default=0, choices=[0,1,2,3], 
                        help='dataset splited configuration: default=0')
    parser.add_argument('--nw', '--num_workers', type=int, default=0, 
                        help='Number of workers used in dataloading: default=0')
    parser.add_argument('--bs', '--batch_size', type=int, default=1, 
                        help='the size of minibatch: default=1')
    parser.add_argument('--optim', type=str, default='adam', 
                        help='which optimizer to be used: default=adam')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate: default=0.0001')
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
                        help='number of steps for saving the model parameters: 50')
    return parser.parse_args() 

def train_model():
    # prepare the data
    train_set = BreakfastDataset(mode="train", split_idx=args.split_idx, preproc=None, over_write=False)
    val_set = BreakfastDataset(mode="val", split_idx=args.split_idx, preproc=None, over_write=False)
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    val_dataloader = DataLoader(dataset=val_set, batch_size=args.bs, shuffle=True, num_workers=args.nw)
    dataset = {"train": train_set, "val": val_set}
    dataloader = {"train": train_dataloader, "val": val_dataloader}
    phase_list = ["train", 'val']
    print("Preparing data done!!!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on {}.".format(device))

    # prepare the model
    model = Anticipation(use_dec=args.use_dec)
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
    # rec_criterion = nn.BCEWithLogitsLoss()
    rec_criterion = nn.CrossEntropyLoss()
    
    #set up logger
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.ds + '/' + args.exp_ver)
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.ds, args.exp_ver), recursive=True)
    
    # start training
    for epoch in range(args.start_epoch, args.epoch):
        loss_list = []
        for phase in phase_list:
            s_t = time.time()
            epoch_loss = 0
            # count = 0
            for data in tqdm(dataloader[phase]):
                # import ipdb; ipdb.set_trace()
                # count += 1; 
                # if count > 10: break
                feat = data[0][:,0:2]
                labels = data[1][:,0:2]
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
                epoch_loss += loss.item() * (labels.shape.numel() - pad_num.sum()).float()
            epoch_loss /= len(dataset[phase])
            loss_list.append(epoch_loss)
            # print loss
            if epoch == 0 or (epoch % args.print_every) == 9:
                e_t = time.time()
                print(f"Phase:[{phase}] Epoch:[{epoch+1}/{args.epoch}]  Loss:[{epoch_loss}]  Execution_time:[{round(e_t-s_t, 1)}]")
        # plot loss
        assert len(phase_list) == len(loss_list)
        if len(phase_list) == 2:
            writer.add_scalars('train_val_loss', {'train': loss_list[0], 'val': loss_list[1]}, epoch)
        else:
            writer.add_scalars('trainval_loss', {'trainval': loss_list[0]}, epoch)
        # save training information and checkpoint
        if epoch % args.save_every == (args.save_every - 1) and epoch >= 0:
            opts = {'lr': args.lr, 'b_s': args.bs, 'optim': args.optim, 'use_dec': args.use_dec}
            save_info = {"arguments": opts, "config": BF_CONFIG}
            io.dumps_json(save_info, os.path.join(args.save_dir, args.ds, args.exp_ver, 'training_info.json'))
            save_name = "checkpoint_" + str(epoch+1) + "_epoch.pth"
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.ds, args.exp_ver, save_name))

    writer.close()
    print("Training finished!!!")
        

if __name__ == "__main__":
    args = arg_parse()
    train_model()
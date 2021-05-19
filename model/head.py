import torch
import torch.nn as nn
import torch.nn.functional as F

class I3D_Recognition_Head(nn.Module):
    def __init__(self, action_num, in_dim=1024):
        super(I3D_Recognition_Head, self).__init__()
        self.avg = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
        self.linear = nn.Linear(in_dim, action_num)
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.avg(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = x.squeeze(2)
        logits = self.linear(x)
        return logits


class I3D_Head(nn.Module):
    def __init__(self, d_in=1024, drop_prob=0.1, use_fc=True, use_lnorm=False):
        super(I3D_Head, self).__init__()
        self.use_fc = use_fc
        self.use_lnorm = use_lnorm
        if use_fc == 1:
            # self.avg = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
            self.fc = nn.Linear(1024, d_in, bias=True)
            self.dropout = nn.Dropout(drop_prob)
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        if use_fc == 2:
            self.fc = nn.Linear(1024, d_in, bias=True)
            self.dropout1 = nn.Dropout(drop_prob)
            self.fc2 = nn.Linear(d_in, d_in, bias=True)
            self.dropout2 = nn.Dropout(drop_prob)
            if use_lnorm:
                self.layer_norm1 = nn.LayerNorm(d_in, eps=1e-6)
                self.layer_norm2 = nn.LayerNorm(d_in, eps=1e-6)
    
    def forward(self, x):
        # x = self.avg(x)
        # x = F.normalize(x)
        if self.use_fc == 1:
            x = self.fc(x)
            x = self.layer_norm(x)
            x = F.relu(x, inplace=True)
            x = self.dropout(x)
            return x
        elif self.use_fc == 2:
            x = self.fc(x)
            if self.use_lnorm:
                x = self.layer_norm1(x)
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            if self.use_lnorm:
                x = self.layer_norm2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            return x
        else:
            return x


class Encoder_Head(nn.Module):
    def __init__(self, action_num, in_dim=1024, layers=1, drop_prob=0.1):
        super(Encoder_Head, self).__init__()
        self.layers = layers
        if layers == 2:
            self.recog_fc1 = nn.Linear(in_dim, in_dim)
            self.dropout = nn.Dropout(drop_prob)
            self.recog_fc2 = nn.Linear(in_dim, action_num)
        else:
            self.recog_fc = nn.Linear(in_dim, action_num)
            # self.recog_fc_time = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.layers==2:
            logits = F.relu(self.recog_fc1(x))
            logits = self.dropout(logits)
            logits = self.recog_fc2(logits)
        else:
            logits = self.recog_fc(x)
            # recog_t = self.recog_fc_time(x)
            # recog_t = F.relu(recog_t)
            recog_t = 0

        return logits, recog_t


class Decoder_Head(nn.Module):
    def __init__(self, action_num, in_dim=1024, layers=1, drop_prob=0.1):
        super(Decoder_Head, self).__init__()
        self.layers = layers
        if layers == 2:
            self.anti_fc1 = nn.Linear(in_dim, in_dim)
            self.dropout = nn.Dropout(drop_prob)
            self.anti_fc2 = nn.Linear(in_dim, action_num)
        else:
            self.anti_fc = nn.Linear(in_dim, action_num)
            # self.anti_fc_time = nn.Linear(in_dim, 1)

    def forward(self, x):
        if self.layers==2:
            logits = F.relu(self.anti_fc1(x))
            logits = self.dropout(logits)
            logits = self.anti_fc2(logits)
        else:
            logits = self.anti_fc(x)
            anti_t = 0
            # anti_t = self.anti_fc_time(x)
            # anti_t = F.relu(anti_t)

        return logits, anti_t

class Decoder_Queries_Gen(nn.Module):
    def __init__(self, in_dim=1024, drop_prob=0.1, all_zeros=False):
        super(Decoder_Queries_Gen, self).__init__()
        self.all_zeros = all_zeros
        if not all_zeros:
            self.fc = nn.Linear(1, in_dim)
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, bs, x):
        # import ipdb; ipdb.set_trace()
        if not self.all_zeros:
            x = x.expand(bs, x.shape[1])[:,:,None]
            output = F.relu(self.fc(x))
            return self.dropout(output)
        else:
            return x

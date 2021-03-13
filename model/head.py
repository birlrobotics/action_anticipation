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
    def __init__(self, d_in=1024, drop_prob=0.1, use_fc=True):
        super(I3D_Head, self).__init__()
        self.use_fc = use_fc
        if use_fc:
            # self.avg = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
            self.fc = nn.Linear(1024, d_in, bias=True)
            self.dropout = nn.Dropout(drop_prob)
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    
    def forward(self, x):
        # x = self.avg(x)
        # import ipdb; ipdb.set_trace()
        # x = F.normalize(x)
        if self.use_fc:
            x = self.fc(x)
            # x = self.layer_norm(x)
            x = F.relu(x)
            return self.dropout(x)
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

    def forward(self, x):
        if self.layers==2:
            logits = F.relu(self.recog_fc1(x))
            logits = self.dropout(logits)
            logits = self.recog_fc2(logits)
        else:
            logits = self.recog_fc(x)

        return logits


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

    def forward(self, x):
        if self.layers==2:
            logits = F.relu(self.anti_fc1(x))
            logits = self.dropout(logits)
            logits = self.anti_fc2(logits)
        else:
            logits = self.anti_fc(x)

        return logits

class Decoder_Queries_Gen(nn.Module):
    def __init__(self, in_dim=1024, drop_prob=0.1):
        super(Decoder_Queries_Gen, self).__init__()
        self.fc = nn.Linear(1, in_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, bs, x):
        x = x.expand(bs, x.shape[1])[:,:,None]
        output = F.relu(self.fc(x))

        return self.dropout(output)

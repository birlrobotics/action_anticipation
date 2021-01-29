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
        # output = self.softmax(logits)
        # return self.softmax(logits)
        return logits


class I3D_Head(nn.Module):
    def __init__(self, d_in=1024, drop_prob=0.1):
        super(I3D_Head, self).__init__()
        # self.avg = nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)
        self.fc = nn.Linear(d_in, d_in, bias=True)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        # x = self.avg(x)
        # x = x.squeeze(2)
        # x = x.squeeze(2)
        # x = x.squeeze(2)
        x = F.relu(self.fc(x))
        return self.dropout(x)


class Encoder_Head(nn.Module):
    def __init__(self, action_num, in_dim=1024):
        super(Encoder_Head, self).__init__()
        self.recog_fc = nn.Linear(in_dim, action_num)

    def forward(self, x):
        logits = self.recog_fc(x)

        return logits


class Decoder_Head(nn.Module):
    def __init__(self, ):
        super(Decoder_Head, self).__init__()

    def forward(self, x):
        return x
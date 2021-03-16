''' Transformer: This code is mainly borrowed from 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch  
    Actually, there is another implementation in Pytorch
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pad_mask(seq_len, pad_num):
    pad_mask = None
    for i in pad_num:
        i = int(i)
        temp = torch.ones(seq_len)
        # to avoid select all elements when i==0
        if i:
            temp[-i:] = 0
        pad_mask = torch.cat((pad_mask, temp[None,:])) if isinstance(pad_mask, torch.Tensor) else temp[None,:]
    return pad_mask.unsqueeze(1) == 1
    # return pad_mask.unsqueeze(1).byte()

def get_sequence_mask(seq_len):
    subseq_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1))
    return subseq_mask == 1
    # return subseq_mask.byte()

def multi_scale_mask(seq_len, h_step_list=[4, 8, 16], h_num_each=2):
    ms_mask = torch.empty((0, seq_len, seq_len))
    for step in h_step_list:
        mask = torch.eye(seq_len)
        for i in range(seq_len):
            l_n = int(step / 2); r_n = int(step / 2)
            if i < seq_len/2:
                for j in range(l_n):
                    if i-j-1 >= 0 and l_n:
                        l_n -= 1
                        mask[i][i-j-1]=1
                    else:
                        break
                r_n += l_n
                for j in range(r_n):
                    mask[i][i+j+1]=1
            else:
                for j in range(r_n):
                    if i+j+1 < seq_len and r_n:
                        r_n -= 1
                        mask[i][i+j+1]=1
                    else:
                        break
                l_n += r_n
                for j in range(l_n):
                    mask[i][i-j-1]=1
        mask = mask[None,:,:].expand((h_num_each, mask.size(0), mask.size(1)))
        ms_mask = torch.cat((ms_mask, mask))
    return ms_mask.byte()

class Transformer(nn.Module):
    def __init__(self, n_layers, n_attn_head, d_input, d_inner, d_qk, d_v, drop_prob=0.1, max_len=300, pos_enc=True, use_dec=True, return_attn=True, msm=False):
        super(Transformer, self).__init__()
        
        self.n_attn_head=n_attn_head
        self.msm = msm
        self.use_dec = use_dec
        self.return_attn = return_attn
        self.encoder = Encoder(n_layers=n_layers, n_attn_head=n_attn_head, d_input=d_input, 
                               d_inner=d_inner, d_qk=d_qk, d_v=d_v, max_len=max_len, drop_prob=drop_prob, pos_enc=pos_enc)
        if use_dec:
            self.decoder = Decoder(n_layers=n_layers, n_attn_head=n_attn_head, d_input=d_input, 
                               d_inner=d_inner, d_qk=d_qk, d_v=d_v, max_len=max_len, drop_prob=drop_prob, pos_enc=pos_enc)
        if msm:
            self.ms_mask = multi_scale_mask(max_len, h_step_list=[4, 8, 16], h_num_each=2)

    def forward(self, src_seq, trg_seq, enc_pad_num, dec_pad_num):
        # TODO: make the mask matrix
        src_mask = None; trg_mask = None
        # import ipdb; ipdb.set_trace()
        src_mask = get_pad_mask(src_seq.shape[1], enc_pad_num).to(src_seq.device)
        # trg_mask = (get_pad_mask(trg_seq.shape[1], dec_pad_num) & get_sequence_mask(trg_seq.shape[1])).to(src_seq.device)
        trg_mask = get_pad_mask(trg_seq.shape[1], dec_pad_num).to(src_seq.device)
        src_mask = src_mask.unsqueeze(1); trg_mask = trg_mask.unsqueeze(1)
        if self.msm:
            self.ms_mask = self.ms_mask.to(src_seq.device)
            exp_shape = (src_mask.size(0), self.n_attn_head-self.ms_mask.shape[0], src_mask.size(-1), src_mask.size(-1))
            src_mask = torch.cat((self.ms_mask & src_mask, src_mask.expand(exp_shape)), dim=1)
            trg_mask = torch.cat((self.ms_mask & trg_mask, trg_mask.expand(exp_shape)), dim=1)
        enc_output, enc_slf_attn_list = self.encoder(src_seq, src_mask, return_attn=self.return_attn)
        if self.use_dec:
            dec_output, dec_slf_attn_list, dec_enc_attn_list = self.decoder(trg_seq, enc_output, src_mask, trg_mask, enc_pad_num, return_attn=self.return_attn) 
            return enc_output, dec_output, enc_slf_attn_list, dec_slf_attn_list, dec_enc_attn_list
        else:
            return enc_output, enc_slf_attn_list


class Encoder(nn.Module):
    def __init__(self, n_layers, n_attn_head, d_input, d_inner, d_qk, d_v, max_len, drop_prob=0.1, pos_enc=True):
        super(Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_input=d_input, max_len=max_len, drop_prob=drop_prob, pos_enc=pos_enc)
        self.encoder_stack = nn.ModuleList([EncoderLayer(n_attn_head, d_input, d_inner, d_qk, d_v, drop_prob) for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)

    def forward(self, x, src_mask, return_attn = False):
        enc_slf_attn_list = []

        enc_output = self.position_enc(x)
        for enc_layer in self.encoder_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attn else []
        
        return enc_output, enc_slf_attn_list


class Decoder(nn.Module):
    def __init__(self, n_layers, n_attn_head, d_input, d_inner, d_qk, d_v, max_len, drop_prob=0.1, pos_enc=True):
        super(Decoder, self).__init__()
        self.position_enc = PositionalEncoding(d_input=d_input, max_len=max_len, drop_prob=drop_prob, pos_enc=pos_enc)
        self.decoder_stack = nn.ModuleList([DecoderLayer(n_attn_head, d_input, d_inner, d_qk, d_v, drop_prob) for _ in range(n_layers)])

    def forward(self, x, enc_output, src_mask, trg_mask, enc_pad_num, return_attn=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.position_enc(x)
        for dec_layer in self.decoder_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []
        
        return dec_output, dec_slf_attn_list, dec_enc_attn_list


class PositionalEncoding(nn.Module):
    '''https://github.com/pytorch/examples/blob/master/word_language_model/model.py'''
    def __init__(self, d_input, max_len, drop_prob=0.1, pos_enc=True):
        super(PositionalEncoding, self).__init__()
        if pos_enc:
            pe = torch.zeros(max_len, d_input)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_input, 2).float() * (-math.log(10000.0) / d_input))
            pe[:, 0::2] = torch.sin(position * div_term) # dim 2i
            pe[:, 1::2] = torch.cos(position * div_term) # dim 2i+1
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.dropout = nn.Dropout(drop_prob)
        self.pos_enc = pos_enc

    def forward(self, x):
        if self.pos_enc:
            x = x + self.pe[:, :x.size(1), :]
            # return self.dropout(x)
            return x
        else:
            return x

class EncoderLayer(nn.Module):
    def __init__(self, n_attn_head, d_input, d_inner, d_qk, d_v, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_attn_head, d_input, d_qk, d_v, drop_prob=drop_prob)
        self.ffn = FeedForward(d_input, d_inner, drop_prob=drop_prob)

    def forward(self, enc_input, slf_attn_mask):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self,  n_attn_head, d_input, d_inner, d_qk, d_v, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_attn_head, d_input, d_qk, d_v, drop_prob=drop_prob)
        self.enc_attn = MultiHeadAttention(n_attn_head, d_input, d_qk, d_v, drop_prob=drop_prob)
        self.ffn = FeedForward(d_input, d_inner, drop_prob=drop_prob)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_attn_head, d_input, d_qk, d_v, drop_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_attn_head = n_attn_head
        self.d_qk = d_qk
        self.d_v = d_v
        
        self.fc_qs = nn.Linear(d_input, n_attn_head * d_qk, bias=False)
        self.fc_ks = nn.Linear(d_input, n_attn_head * d_qk, bias=False)
        self.fc_vs = nn.Linear(d_input, n_attn_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature = d_qk ** 0.5, attn_drop_prob=0)

        self.fc = nn.Linear(n_attn_head * d_v, d_input, bias=False)
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)

    def forward(self, queries, keys, values, mask=None):
        sz_b, len_q, len_k, len_v = queries.size(0), queries.size(1), keys.size(1), values.size(1)
        # pass through the pre-attention projection layer 
        # and separate the output into different heads: (b_size, seq_len, (n*d)) --> (b_size, seq_len, n, d) 
        # TODO: shall we need the activation function 
        q = self.fc_qs(queries).view(sz_b, len_q, self.n_attn_head, self.d_qk)
        k = self.fc_ks(keys).view(sz_b, len_k, self.n_attn_head, self.d_qk)
        v = self.fc_vs(values).view(sz_b, len_v, self.n_attn_head, self.d_v)

        # transpose for attention dot product: --> (b_size, n, seq_len, d) 
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # for head axis broadcasting
        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += queries
        output = self.layer_norm(output)

        return output, attn 


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_drop_prob=0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_drop_prob)

    def forward(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q / self.temperature, k.transpose(2,3))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
                
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, v)

        return output, attn_probs


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.dropout1 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.dropout2 = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        
    def forward(self, x):
        output = self.dropout1(F.relu(self.fc1(x)))
        output = self.dropout2(self.fc2(output))
        output += x
        output = self.layer_norm(output)

        return output


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError("activation should be relu/gelu/prelu, not {}".format(activation))

# def get_activation(name):
#     if name=='ReLU':
#         return nn.ReLU(inplace=True)
#     elif name=='Tanh':
#         return nn.Tanh()
#     elif name=='Identity':
#         return Identity()
#     elif name=='Sigmoid':
#         return nn.Sigmoid()
#     elif name=='LeakyReLU':
#         return nn.LeakyReLU(0.2,inplace=True)
#     else:
#         assert(False), 'Not Implemented'
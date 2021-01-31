import torch
import torch.nn as nn
import torchvision.models._utils as _utils
from model.i3d import I3D
from model.transformer import Transformer
from model.head import I3D_Head, Encoder_Head, Decoder_Head, Decoder_Queries_Gen
from dataset.config import BF_CONFIG, BF_ACTION_CLASS

# TODO: change the name
class Anticipation_With_Backbone(nn.Module):
    def __init__(self, use_dec=True):
        super(Anticipation_With_Backbone, self).__init__()
        self.use_dec = use_dec
        backbone = self._make_backbone(name=BF_CONFIG["backbone"], ck_dir=BF_CONFIG["cp_dir"], fixed=BF_CONFIG["fixed"])
        self.backbone = _utils.IntermediateLayerGetter(backbone, BF_CONFIG['RETURN_LAYERS'])
        self.i3d_head = I3D_Head(d_in=BF_CONFIG["d_input"], drop_prob=BF_CONFIG["drop_prob"])
        self.transformer = Transformer(BF_CONFIG["n_layers"], BF_CONFIG["n_attn_head"], BF_CONFIG["d_input"], 
                                       BF_CONFIG["d_inner"], BF_CONFIG["d_qk"], BF_CONFIG["d_v"], BF_CONFIG["drop_prob"], 
                                       BF_CONFIG["video_len"], use_dec=use_dec, pos_enc=BF_CONFIG["pos_enc"], return_attn=BF_CONFIG["return_attn"])
        self.enc_head = Encoder_Head(len(BF_ACTION_CLASS), in_dim=BF_CONFIG["d_input"])
        self.dec_head = Decoder_Head()

        # initialization
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs_feat, queries, enc_pad_num, dec_pad_num):
        backbone_feat = None
        for i in range(x.shape[0]):
            temp_feat = self.backbone(obs_feat[i])
            temp_feat = self.i3d_head(temp_feat['feat'])
            backbone_feat = torch.cat((backbone_feat, temp_feat[None, :])) if isinstance(backbone_feat, torch.Tensor) else temp_feat[None, :]
        if self.use_dec:
            enc_output, dec_output, *attn = self.transformer(obs_feat, queries, enc_pad_num, dec_pad_num)
            recog_logits = self.enc_head(enc_output)
            antic_logits = self.dec_head(dec_output)
            return recog_logits, antic_logits, attn
        else:
            enc_output,*attn = self.transformer(obs_feat, obs_feat, enc_pad_num, enc_pad_num)
            recog_logits = self.enc_head(enc_output)
            return recog_logits, attn         

    def _make_backbone(self, name='i3d', ck_dir=None, fixed=True):
        model = I3D()
        checkpoints = torch.load(ck_dir)
        model.load_state_dict(checkpoints)
        if fixed:
            for param in model.parameters():
                param.requires_grad = False
        return model.features

class Anticipation_Without_Backbone(nn.Module):
    def __init__(self, use_dec=True):
        super(Anticipation_Without_Backbone, self).__init__()
        self.use_dec = use_dec
        self.i3d_head = I3D_Head(d_in=BF_CONFIG["d_input"], drop_prob=BF_CONFIG["drop_prob"])
        self.queries_gen = Decoder_Queries_Gen(in_dim=BF_CONFIG["d_input"], drop_prob=BF_CONFIG["drop_prob"])
        self.transformer = Transformer(BF_CONFIG["n_layers"], BF_CONFIG["n_attn_head"], BF_CONFIG["d_input"], 
                                       BF_CONFIG["d_inner"], BF_CONFIG["d_qk"], BF_CONFIG["d_v"], BF_CONFIG["drop_prob"], 
                                       BF_CONFIG["video_len"], use_dec=use_dec, pos_enc=BF_CONFIG["pos_enc"])
        self.enc_head = Encoder_Head(len(BF_ACTION_CLASS), in_dim=BF_CONFIG["d_input"], drop_prob=BF_CONFIG["drop_prob"])
        self.dec_head = Decoder_Head(len(BF_ACTION_CLASS), in_dim=BF_CONFIG["d_input"], drop_prob=BF_CONFIG["drop_prob"])

        # initialization
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs_feat, queries, enc_pad_num, dec_pad_num):
        backbone_feat = self.i3d_head(obs_feat)
        if self.use_dec:
            # import ipdb; ipdb.set_trace()
            queries = self.queries_gen(obs_feat.shape[0], queries)
            enc_output, dec_output, *attn = self.transformer(backbone_feat, queries, enc_pad_num, dec_pad_num)
            recog_logits = self.enc_head(enc_output)
            antic_logits = self.dec_head(dec_output)
            return recog_logits, antic_logits, attn
        else:
            enc_output,*attn = self.transformer(backbone_feat, backbone_feat, enc_pad_num, dec_pad_num)
            recog_logits = self.enc_head(enc_output)
            return recog_logits, attn   
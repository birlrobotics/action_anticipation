import torch

class LabelSmoothingLoss:
    def __init__(self, eps=0.1, ignore_index=-100, reduction='mean'):
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def __call__(self, pred_logits, gt_labels):
        num_class = pred_logits.shape[-1]
        mask = gt_labels.ne(self.ignore_index)
        pred_logits, gt_labels = pred_logits.masked_select(mask[:, None]).view(-1, num_class), gt_labels.masked_select(mask)        
        pred_prob = pred_logits.log_softmax(dim=-1)

        one_hot = torch.zeros_like(pred_prob).scatter(1, gt_labels[:, None], 1)
        one_hot_smooth = one_hot * (1 - self.eps) + (1 - one_hot)*self.eps / (num_class - 1)

        loss = -(one_hot_smooth * pred_prob).sum(-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            pass
        return loss


class Mulit_Loss():
    def __init__(self, eps=0, ignore_index=-100, reduction='mean'):
        self.reduction = reduction
        if eps:
            self.recog_criterion = LabelSmoothingLoss(eps=eps, ignore_index=ignore_index, reduction=reduction)
            self.anti_criterion = LabelSmoothingLoss(eps=eps, ignore_index=ignore_index, reduction=reduction)
        else:
            self.recog_criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)    # 'mean' 'sum' 'none'
            self.anti_criterion = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def _time_loss(self, pred, gt):
        norm_factor = 30
        smooth_facotr = 1.5
        loss_list = torch.pow(gt/norm_factor - pred, 2)
        loss_list = torch.where(loss_list <= (smooth_facotr/norm_factor)**2, torch.tensor(0.0).to(pred.device), loss_list)
        if self.reduction == 'mean':
            loss = torch.mean(loss_list)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_list)
        else:
            loss = loss_list
        return loss

    def __call__(self, recog_logits, anti_logits, recog_itval, anti_itval, obs_labels, anti_labels, obs_itval_gt, anti_itval_gt):
        # import ipdb; ipdb.set_trace()
        recog_loss = self.recog_criterion(recog_logits.reshape(recog_logits.shape[:-1].numel(), recog_logits.shape[-1]), obs_labels.reshape(obs_labels.shape.numel()))
        anti_loss = self.anti_criterion(anti_logits.reshape(anti_logits.shape[:-1].numel(), anti_logits.shape[-1]), anti_labels.reshape(anti_labels.shape.numel()))
        # recog_t_loss = self._time_loss(recog_itval, obs_itval_gt)
        # anti_t_loss = self._time_loss(anti_itval, anti_itval_gt)
        recog_t_loss, anti_t_loss = 0, 0

        return recog_loss, anti_loss, recog_t_loss, anti_t_loss


class WarmUpOptimizer():
    def __init__(self, optimizer, d_model, lr_factor, warmup_step, step=0):
        self.optimizer = optimizer
        self.constant = d_model ** (-0.5)
        self.lr_factor = lr_factor
        self.warmup_constant = warmup_step ** (-1.5)
        self.cur_step = step

    def step(self):
        self.cur_step += 1
        lr = self.update_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.optimizer.step()

    def update_lr(self):
        return self.lr_factor * (self.constant * min(self.cur_step ** (-0.5), self.cur_step * self.warmup_constant))

    def zero_grad(self):
        self.optimizer.zero_grad()


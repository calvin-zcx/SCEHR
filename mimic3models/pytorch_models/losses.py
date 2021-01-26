"""
Author for SCL: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Others: Chengxi Zang
Data: 2021-01-18
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # (128,128)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (128,128)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  #(128)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupNCELoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1., contrast_mode='all'):  # base_temperature=0.07):
        super(SupNCELoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask[mask==0] = -1.
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # mask = mask * logits_mask

        log_pos_neg = -logits_mask * torch.log(torch.sigmoid(mask * anchor_dot_contrast)) #.log() #.sum(1, keepdim=True)
        loss = log_pos_neg.mean(1, keepdim=True)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def CBCE_WithLogitsLoss_separate(wx_pos, wx_neg, y): #, is_stable=True):
    # contrastive binary cross entropy loss
    # wx_pos: (bs, 1), before sigmoid
    # wx_neg: (bs, 1), before sigmoid
    # y: (bs,)
    pos_part = F.binary_cross_entropy_with_logits(wx_pos, y)
    neg_part = F.binary_cross_entropy_with_logits(wx_neg, 1.-y)
    ret = (pos_part + neg_part)/2.0
    return ret


def CBCE_WithLogitsLoss(wx, y):
    # wx: (bs, 2), before sigmoid
    # y: (bs,)
    return CBCE_WithLogitsLoss_separate(wx[:, 1], wx[:, 0], y)

#
# def CBCE_WithLogitsLoss(wx, y): #, is_stable=True):
#     # contrastive binary cross entropy loss
#     # wx: (bs, 2), before sigmoid
#     # y: (bs,)
#
#     pos_part = F.binary_cross_entropy_with_logits(wx[:, 1], y)
#     neg_part = F.binary_cross_entropy_with_logits(wx[:, 0], 1.-y)
#     ret = (pos_part + neg_part)/2.0
#     return ret


def CBCE_loss_separate(y_pre_pos, y_pre_neg, y):
    # contrastive binary cross entropy loss
    # y_pre_pos: (bs, 1), after sigmoid
    # y_pre_neg: (bs, 1), after sigmoid
    # y: (bs,)
    i1 = (y == 1)
    i0 = (y == 0)
    pos_anchor = y_pre_pos[i1].log() + (1. - y_pre_neg[i1]).log()
    neg_anchor = y_pre_neg[i0].log() + (1. - y_pre_pos[i0]).log()
    ret = -torch.cat([pos_anchor, neg_anchor]).mean()/2.0
    return ret


def CBCE_loss(y_pre, y):
    # y_pre: (bs, 2), after sigmoid
    # y: (bs,)
    return CBCE_loss_separate(y_pre[:, 1], y_pre[:, 0], y)


# def CBCE_loss(y_pre, y):
#     # contrastive binary cross entropy loss
#     # y_pre: (bs, 2), after sigmoid
#     # y: (bs,)
#     i1 = (y == 1)
#     i0 = (y == 0)
#     pos_anchor = y_pre[i1, 1].log() + (1. - y_pre[i1, 0]).log()
#     neg_anchor = y_pre[i0, 0].log() + (1. - y_pre[i0, 1]).log()
#     # pos_anchor = (y_pre[i1, 1] * (1. - y_pre[i1, 0])).log()
#     # neg_anchor = (y_pre[i0, 0] * (1. - y_pre[i0, 1])).log()
#     ret = -torch.cat([pos_anchor, neg_anchor]).mean()/2.0
#     return ret


def CBCE_loss_multilabel(y_pre, y):
    # x: (bs, 2*C), after sigmoid
    # y: (bs, C)
    assert len(y.shape) == 2
    losses = []
    y_pre_pos, y_pre_neg = torch.chunk(y_pre, 2, dim=1)
    C = y_pre_pos.shape[1]
    for c in range(C):
        ret = CBCE_loss_separate(y_pre_pos[:, c], y_pre_neg[:, c], y[:,c])
        losses.append(ret)
    loss = torch.mean(torch.stack(losses))
    return loss


def CBCE_WithLogitsLoss_multilabel(wx, y):
    # wx: (bs, 2*C), before sigmoid
    # y: (bs, C)
    assert len(y.shape) == 2
    losses = []
    wx_pos, wx_neg = torch.chunk(wx, 2, dim=1)
    C = wx_pos.shape[1]
    for c in range(C):
        ret = CBCE_WithLogitsLoss_separate(wx_pos[:, c], wx_neg[:, c],  y[:, c])
        losses.append(ret)
    loss = torch.mean(torch.stack(losses))
    return loss


# def criterion_SCL_multilabel(SCL, features, labels):
#     assert len(labels.shape) == 2
#     losses = []
#     for j in range(labels.shape[1]):
#         y = labels[:, j]
#         if y.sum().item() > 1:
#             losses.append(SCL(features, y))
#     r = torch.mean(torch.stack(losses))
#     return r


class SupConLoss_MultiLabel(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_MultiLabel, self).__init__()
        self.scl = SupConLoss(temperature, contrast_mode, base_temperature)
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        assert len(labels.shape) == 2
        losses = []
        for j in range(labels.shape[1]):
            y = labels[:, j]
            if y.sum().item() > 1:
                losses.append(self.scl(features, y))
            else:
                print('Warning: #Positive < 2 for {}^th dim label. '
                      'Not Supervised Contrastive Regularizer for this batch'.format(j))
        l = torch.mean(torch.stack(losses))
        return l


def test_CBCE_loss(n=5):
    wx = torch.randn(n, 2)
    y = torch.empty(n).random_(2)
    y_pre = torch.sigmoid(wx)
    l1 = CBCE_loss(y_pre, y)
    l2 = CBCE_WithLogitsLoss(wx, y)
    return l1, l2


def test_CBCE_multilabel_loss(n=5, c=2):
    wx = torch.randn(n, c*2)
    y = torch.empty(n, c).random_(2)
    y_pre = torch.sigmoid(wx)
    l1 = CBCE_loss_multilabel(y_pre, y)
    l2 = CBCE_WithLogitsLoss_multilabel(wx, y)
    return l1, l2
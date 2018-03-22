from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self, attr_weight=0.01):
        super(LanguageModelCriterion, self).__init__()
        self.seen = False
        self.attr_cr = nn.CrossEntropyLoss()
        self.attr_weight = attr_weight

    def forward(self, model_output, target, mask, attr):

        pred_seq, pred_attr = model_output

        # input (from model.forward())      (batch_size, max_seq_len, vocab_size)
        # target (from dataloader->labels)  (batch_size, max_seq_len)
        # mask (from dataloader->masks)     (batch_size, max_seq_len)

        if not self.seen:
            print('> in LanguageModelCriterion.forward(input, target, mask):')
            print('    pred_seq', pred_seq.shape)  # (200, 17, 3562)
            print('    pred_attr', pred_attr.shape)  # (200, 1000)
            print('    target', target.shape)  # (200, 17)
            print('    mask', mask.shape)  # (200, 17)
            print('    attr', attr.shape)  # (200, 1000)
            self.seen = True

        # truncate to the same size
        target = target[:, :pred_seq.size(1)]
        mask =  mask[:, :pred_seq.size(1)]
        pred_seq = to_contiguous(pred_seq).view(-1, pred_seq.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - pred_seq.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        bsize = pred_attr.size(0)
        pred_attr = to_contiguous(pred_attr)
        attr = to_contiguous(attr.float())
        attr_loss = torch.pow(torch.sum(torch.pow((pred_attr - attr), 2)), 0.5) / bsize

        output = output + self.attr_weight * attr_loss

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import math

def language_eval(dataset, preds, model_id, split):
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif 'msvd' in dataset:
        sys.path.append('coco-caption')
        annFile = 'coco-caption/annotations/coco_ref_msvd.json'
    elif 'kuaishou' in dataset:
        sys.path.append('coco-caption')
        annFile = 'coco-caption/annotations/coco_ref_kuaishou.json'
    else:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    print_all_beam = eval_kwargs.get('print_all_beam', False)

    print('> print_all_beam', print_all_beam)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['attributes']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, attributes = tmp

            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:], attributes).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        seq, prob, attr = model.sample(fc_feats, att_feats, eval_kwargs)

        if print_all_beam is True:
            for p in xrange(seq.shape[0]):
                seq_this = seq[p, :, :]
                prob_this = prob[p, :, :]
                sents = utils.decode_sequence(loader.get_vocab(), seq_this)

                print('---------------------------------------------------------')
                print('> video id %s:' % data['infos'][p]['id'])

                for k, sent in enumerate(sents):
                    entry = {'image_id': data['infos'][p]['id'], 'caption': sent}
                    if eval_kwargs.get('dump_path', 0) == 1:
                        entry['file_name'] = data['infos'][p]['file_path']
                    predictions.append(entry)
                    if eval_kwargs.get('dump_images', 0) == 1:
                        # dump the raw image to vis/ folder
                        cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                        print(cmd)
                        os.system(cmd)

                    if verbose:
                        print('    %s (%.5f)' %(entry['caption'], math.exp(sum(prob_this[k, :]))))

                print('---------------------------------------------------------')
                if split == 'show':
                    p = raw_input()
            # seq [image_idx, beam_idx, sentence]
        else:
            
            #set_trace()
            sents = utils.decode_sequence(loader.get_vocab(), seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                if eval_kwargs.get('dump_path', 0) == 1:
                    entry['file_name'] = data['infos'][k]['file_path']
                predictions.append(entry)
                if eval_kwargs.get('dump_images', 0) == 1:
                    # dump the raw image to vis/ folder
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                    print(cmd)
                    os.system(cmd)

                this_attr = attr[k, :].data.cpu().numpy()
                assert this_attr.shape == (1000,)

                this_gt_attr = attributes[k * loader.seq_per_img, :].data.cpu().numpy()
                gt_attr_indices = this_gt_attr.argsort()[-5:][::-1]

                attr_indices = this_attr.argsort()[-5:][::-1]

                gt_label = labels[k * loader.seq_per_img, 1:].data.cpu().numpy()

                if verbose:
                    print('video %s: %s' % (entry['image_id'], entry['caption']))
                    print('   gt: %s' % ' '.join(([loader.ix_to_word[str(p)] for p in gt_label if p > 0])))
                    print('   attr: %s' % (' '.join([loader.attr_idx2word[id] for id in attr_indices])))
                    print('   gt labels: %s' % (' '.join([loader.attr_idx2word[id] for id in gt_attr_indices])))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose and split is not 'show':
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

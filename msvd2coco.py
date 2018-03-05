from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import json
import random

def read():
    with open('vid2name.pkl') as file:
        vid2name = cPickle.load(file)
    # vid2name: int -> str
    n_video = len(vid2name.keys())
    print('> total video count (n_video): {}'.format(n_video))

    vids = vid2name.keys()
    random.shuffle(vids)

    # traditional msvd split: 1200 | 100 | 670
    train_split = vids[0: 1200]
    valid_split = vids[1200: 1300]
    test_split = vids[1300: ]

    train_dict = {vid : 'train' for vid in train_split}
    valid_dict = {vid : 'val' for vid in valid_split}
    test_dict = {vid : 'test' for vid in test_split}

    split_dict = train_dict
    split_dict.update(valid_dict)
    split_dict.update(test_dict)

    # should construct cap as a dictionary:
    # cap['vid0'] -> list of {'cap_id', 'tokenized'(train), 'caption'(eval), 'image_id'}
    with open('caps.pkl') as file:
        caps = cPickle.load(file)

    # will be dumped as dataset.json
    images = []
    for vid in range(n_video):
        video_dict = {}
        video_dict['imgid'] = vid
        video_dict['cocoid'] = vid
        video_dict['sentences'] = []
        video_dict['sentids'] = []
        video_dict['split'] = split_dict[vid]

        images.append(video_dict)

    # will be dumped as coco_reference.json
    # only contains TEST set captions.
    annotations = []

    sentence_id = 0

    for vid in range(n_video):
        for cap in caps['vid%s'%vid]:
            sent_dict = {}
            sent_dict['imgid'] = vid
            sent_dict['raw'] = cap['caption']
            sent_dict['tokens'] = cap['tokenized'].split(' ')
            sent_dict['sentid'] = sentence_id

            if images[vid]['split'] == 'test':
                anno_dict = {}
                anno_dict['caption'] = cap['caption']
                anno_dict['id'] = sentence_id
                anno_dict['image_id'] = vid
                annotations.append(anno_dict)

            images[vid]['sentences'].append(sent_dict)
            images[vid]['sentids'].append(sentence_id)
            sentence_id += 1

    # dump dataset_msvd.json and coco_reference_msvd.json
    dataset_msvd = {}
    dataset_msvd['images'] = images
    with open('dataset_msvd.json', 'w') as file:
        json.dump(dataset_msvd, file)
    print('> wrote dataset_msvd.json')

    coco_ref_msvd = {}
    coco_ref_msvd['annotations'] = annotations
    with open('coco_ref_msvd.json') as file:
        json.dump(coco_ref_msvd, file)
    print('> wrote coco_ref_msvd.json')


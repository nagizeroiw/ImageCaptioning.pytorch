
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import h5py


def main(params):

    print('loading h5py file')
    h5_label_file = h5py.File(params['input_label_h5'], 'r', driver='core')
    h5_attributes = h5_label_file['label_attribute'][:]
    print(h5_attributes.shape)

    with open(params['input_json'], 'r') as file:
        dataset = json.load(file)

    with open(params['input_attribute_vocab']) as file:
        a_word2idx = json.load(file)
    a_idx2word = {}
    for w, id in a_word2idx.iteritems():
        a_idx2word[id] = w

    vid = params['vid']
    vid = int(vid)

    video_dicts = dataset['images']
    for video in video_dicts:
        if video['cocoid'] == vid:
            video_dict = video
            break

    attrs = ['cocoid', 'imgid', 'split']
    for attr in attrs:
        print('%s\t%s' % (attr, video_dict[attr]))

    a_vec = h5_attributes[vid]
    assert a_vec.shape == (500,)

    print(sum(a_vec))

    for i in range(a_vec.shape[0]):
        if a_vec[i] > 0:
            print('> %d\t%s' % (i, a_idx2word[i]))

    indices = a_vec.argsort()[-5:][::-1]
    print(' '.join([str(p) for p in indices]))

    print(' '.join([a_idx2word[p] for p in indices]))

    for sent in video_dict['sentences']:
        # print(sent['raw'])
        print('->', ' '.join(sent['tokens']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_label_h5', default='./data/msvd_dataset/msvd_label.h5')
    parser.add_argument('--input_json', default='./data/msvd_dataset/dataset_msvd.json', help='same input json as prepro_*.py')
    parser.add_argument('--input_attribute_vocab', default='./data/msvd_dataset/attribute_word2idx.json', help='dumped attribute word2idx')
    parser.add_argument('--vid', default=0, help='video id to be visualized')

    args = parser.parse_args()
    params = vars(args)
    print('> parsed parameters')
    print(json.dumps(params, indent=2))
    main(params)

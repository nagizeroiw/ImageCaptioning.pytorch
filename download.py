import cPickle
import sys
import os

vid2name_path = {
    'msvd': './data/msvd_raw/vid2name.pkl',
    'kuaishou': './data/kuaishou_raw/vid2name.pkl'
}

vid_url = {
    'msvd': 'jungpu2:~/vc/data/msvd_raw/YouTubeClips/',
    'kuaishou': 'jungpu4:~/video_caption/kuaishou_dataset/caption_video_thu/'
}

def main(dataset, vid):
    vid2name = cPickle.load(open(vid2name_path[dataset]))
    os.system('scp %s%s ./clips/%s_%d.%s' % (vid_url[dataset], vid2name[vid], dataset, vid, vid2name[vid].split('.')[-1]))

if __name__ == '__main__':
    dataset = sys.argv[1]
    vid = int(sys.argv[2])
    main(dataset, vid)

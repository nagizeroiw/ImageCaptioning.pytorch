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
    if dataset == 'kuaishou':
        vid_key = str(vid)
        name = vid_url[dataset] + vid2name[vid_key] + '.mp4'
        ex = '.mp4'
    else:
        vid_key = vid
        name = vid_url[dataset] + vid2name[vid_key]
        ex = vid2name[vid_key].split('.')[-1]
    os.system('scp %s ./clips/%s_%d.%s' % (name, dataset, vid, ex))

if __name__ == '__main__':
    dataset = sys.argv[1]
    vid = int(sys.argv[2])
    main(dataset, vid)

ID=msvd_topdown
scp -r jungpu4:~/vc/log_$ID ./
tensorboard --logdir=./log_$ID

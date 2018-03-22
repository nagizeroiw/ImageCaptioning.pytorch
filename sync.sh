ID=fc_attr
rsync -av jungpu1:~/vc/log_$ID/events* ./log_$ID/
tensorboard --logdir=./log_$ID

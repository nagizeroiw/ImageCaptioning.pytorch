ID=hard_attr3+
rsync -av jungpu1:~/vc/log_$ID/events* ./log_$ID/
tensorboard --logdir=./log_$ID

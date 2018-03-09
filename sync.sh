ID=no_decay
scp -r jungpu1:~/vc/log_$ID ./
tensorboard --logdir=./log_$ID
rm -r ./log_$ID

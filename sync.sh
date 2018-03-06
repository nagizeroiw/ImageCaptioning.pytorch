ID=slower_decay
scp -r jungpu4:~/vc/log_$ID ./
tensorboard --logdir=./log_$ID
rm -r ./log_$ID

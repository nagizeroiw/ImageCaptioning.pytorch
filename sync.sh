ID=tch_force
scp -r jungpu4:~/vc/log_$ID ./
tensorboard --logdir=./log_$ID
rm -r ./log_$ID

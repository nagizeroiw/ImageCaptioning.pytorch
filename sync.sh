ID=topdown_test
scp -r jungpu4:~/vc/log_$ID ./
tensorboard --logdir=./log_$ID

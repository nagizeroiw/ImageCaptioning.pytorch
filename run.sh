ID=topdown_test
python train.py --id st --caption_model topdown \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att \
    --input_label_h5 data/cocotalk_label.h5 \
    --batch_size 64 --learning_rate 2e-4 --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_$ID --save_checkpoint_every 2000 \
    --cmd_log_every 100 \
    --val_images_use 2000 --max_epochs 25

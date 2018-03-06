ID=slower_decay
python train.py --id $ID --caption_model topdown \
    --input_json data/msvd_dataset/msvd.json \
    --input_fc_dir data/msvd_dataset/msvd_fc --input_att_dir data/msvd_dataset/msvd_att \
    --input_label_h5 data/msvd_dataset/msvd_label.h5 \
    --batch_size 5 --learning_rate 2e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 5 --learning_rate_decay_rate 0.9 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_$ID --save_checkpoint_every 1500 \
    --cmd_log_every 100 \
    --val_images_use 670 --max_epochs 400

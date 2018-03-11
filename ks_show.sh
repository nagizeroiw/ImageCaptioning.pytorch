ID=no_decay
python show.py --id $ID --caption_model topdown --start_from ./log_$ID \
    --input_json data/kuaishou_dataset/kuaishou.json \
    --input_fc_dir data/kuaishou_dataset/kuaishou_fc --input_att_dir data/kuaishou_dataset/kuaishou_att \
    --input_label_h5 data/kuaishou_dataset/kuaishou_label.h5 \
    --batch_size 5 --learning_rate 2e-4 --learning_rate_decay_start 100 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_$ID --save_checkpoint_every 1500 \
    --cmd_log_every 100 \
    --val_images_use 499 --max_epochs 400

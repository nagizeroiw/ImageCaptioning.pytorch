ID=hard_attr3+
CUDA_VISIBLE_DEVICES=0 python train.py --id $ID --caption_model topdown \
    --input_json data/msvd_dataset/msvd.json \
    --input_fc_dir data/msvd_dataset/msvd_fc --input_att_dir data/msvd_dataset/msvd_att \
    --input_label_h5 data/msvd_dataset/msvd_label.h5 --input_attribute_json data/msvd_dataset/attribute_word2idx.json\
    --learning_rate_decay_start -1 \
    --attr_weight 0.01 --attr_as_lang_input True \
    --batch_size 10 --learning_rate 0.001 \
    --scheduled_sampling_start -1 \
    --checkpoint_path log_$ID --save_checkpoint_every 1500 \
    --cmd_log_every 100 \
    --val_images_use 670 --max_epochs 400

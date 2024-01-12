CUDA_VISIBLE_DEVICES=0 python test.py \
--exp_name $exp_name$ \
--dataset_name mimic_cxr \
--label_path $path_to_label$  \
--max_seq_length 100 \
--threshold 10 \
--batch_size 64 \
--epochs 1 \
--lr_ve 5e-5 \
--lr_ed 1e-4 \
--save_dir results/mimic_cxr \
--ve_name densenet121 \
--ed_name r2gen \
--cfg configs/swin_tiny_patch4_window7_224.yaml \
--pretrained $path_to_trained_model_ \
--early_stop 10 \
--d_vf 1024 \
--weight_decay 5e-5 \
--optim Adam \
--decay_epochs 50 \
--warmup_epochs 3 \
--warmup_lr 5e-6 \
--lr_scheduler step \
--decay_rate 0.8 \
--seed 456789 \
--addcls  \
--cls_w 1 \
--fbl \
--attn_cam \
--topk 0.25 \
--layer_id 2 \
--attn_method max \
--mse_w 0.5 \
--test \
#--early_exit \
#--sub_back \
#--randaug  \
#--finetune  \
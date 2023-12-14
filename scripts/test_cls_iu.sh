CUDA_VISIBLE_DEVICES=0 python test_cls.py \
--exp_name test_cls_iu \
--dataset_name iu_xray \
--label_path labels/labels.json \
--max_seq_length 60 \
--threshold 3 \
--batch_size 32 \
--epochs 30 \
--lr_ve 5e-5 \
--lr_ed 1e-4 \
--save_dir results/iu_xray \
--cfg configs/swin_tiny_patch4_window7_224.yaml \
--pretrained visualizations/vis/iu_dense121_r2gen_1e-3_2e-3_sd2401_cls1_fbl_mse015_lyid2_tp025_ad_fin_bs32/iu_xray.pth \
--ve_name densenet121 \
--ed_name r2gen \
--early_stop 10 \
--d_vf 2048 \
--weight_decay 5e-5 \
--optim Adam \
--decay_epochs 50 \
--warmup_epochs 3 \
--warmup_lr 5e-6 \
--lr_scheduler step \
--decay_rate 0.8 \
--seed 2401 \
--vis \
--addcls  \
--cls_w 1 \
--fbl \
--attn_cam \
--topk 0.25 \
--layer_id 2 \
--attn_method max \
--mse_w 0.15 \

#--early_exit \
#--sub_back \
#--randaug  \
#--finetune  \
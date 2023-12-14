CUDA_VISIBLE_DEVICES=0 python test_cls.py \
--exp_name test_cls_mimic \
--dataset_name mimic_cxr_dsr2 \
--label_path labels/labels_all_one.json \
--max_seq_length 100 \
--threshold 10 \
--batch_size 32 \
--epochs 30 \
--lr_ve 5e-5 \
--lr_ed 1e-4 \
--save_dir results/mimic_cxr1 \
--cfg configs/swin_tiny_patch4_window7_224.yaml \
--pretrained visualizations/vis/mic_ds121_r2gen_5e-5_1e-4_st11_wum3_5e-6_6_cls1_fbl_mse05_top03_lyid2_ad_op_fin_test/mimic_cxr.pth \
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
--seed 456789 \
--vis \
--addcls  \
--cls_w 1 \
--fbl \
--attn_cam \
--topk 0.3 \
--layer_id 2 \
--attn_method max \
--mse_w 0.5 \

#--early_exit \
#--sub_back \
#--randaug  \
#--finetune  \
DATA_PATH_BASE='./imagenet_c'
RESUME_MODEL='checkpoints/mae_pretrain_vit_large_full.pth'
RESUME_FINETUNE='checkpoints/prob_lr1e-3_wd.2_blk12_ep20.pth'
OUTPUT_DIR_BASE='./output/online3'

python online_all.py \
    --data_path "$DATA_PATH_BASE" \
    --model mae_vit_large_patch16 \
    --input_size 224 \
    --batch_size 32 \
    --steps_per_example 10 \
    --mask_ratio 0.75 \
    --blr 1e-2 \
    --norm_pix_loss \
    --optimizer_type 'sgd' \
    --classifier_depth 12 \
    --head_type "vit_head" \
    --single_crop \
    --output_dir "$OUTPUT_DIR_BASE" \
    --dist_url "file://$OUTPUT_DIR_BASE/$TIME" \
    --finetune_mode 'encoder' \
    --resume_model ${RESUME_MODEL} \
    --resume_finetune ${RESUME_FINETUNE} \
    --device cuda:3
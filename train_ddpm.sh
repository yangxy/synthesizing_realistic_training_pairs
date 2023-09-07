TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
    train_ddpm.py \
    --dataset_name did \
    --use_ema \
    --resolution=256 --random_flip \
    --train_batch_size=4 \
    --max_diffusion_step 500 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision="no" \
    --checkpointing_steps=10000 \
    --num_epochs=100 \
    --save_images_epochs=10 \
    --prediction_type='epsilon' \
    --learning_rate=1e-04 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir="runs/ddpm_did" \
    --dataloader_num_workers=64 \
    --lq_folder="datasets/did" \
#   --multi_gpu --num_processes=8 --gpu_ids '0,1,2,3,4,5,6,7' \

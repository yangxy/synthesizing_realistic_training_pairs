import os
import glob
import cv2
import torch
import random
import argparse
import numpy as np
from torchvision import utils
import torch.nn.functional as F

from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, UNet2DModel

from dataloader.realesrgan import RealESRGAN_degradation
from pipelines.pipeline_ddpm import DDPMPipeline
from pipelines.pipeline_ddpm_img2img import DDPMImg2ImgPipeline

def main(args):
    os.makedirs(f"{args.output_dir}/HR", exist_ok=True)
    os.makedirs(f"{args.output_dir}/LR", exist_ok=True)

    unet = UNet2DModel.from_pretrained(args.pretrained_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.cuda()
    scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_path, subfolder="scheduler")
    pipeline = DDPMImg2ImgPipeline(unet=unet, scheduler=scheduler)

    generator = torch.Generator(device=pipeline.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    degradation = RealESRGAN_degradation()
    
    hq_files = sorted(glob.glob(f'{args.hq_folder}/*.*g'))
    for hq_file in hq_files:
        hq_image = cv2.imread(hq_file)[:,:,::-1] # BGR --> RGB
        filename, ext = os.path.splitext(os.path.basename(hq_file))
        GT_image, LR_image_init = degradation.degrade_process(hq_image/255., resize_bak=False)
        utils.save_image(GT_image.flip(1), f"{args.output_dir}/HR/{filename}.png")

        LR_image_init = LR_image_init * 2.0 - 1.0 # [0,1] -> [-1,1]
        # run pipeline in inference (sample random noise and denoise)
        strength = max(random.random() * args.max_strength, 1.0/args.num_inference_steps) # max_diffusion_step is 250=1000*0.25
        try:
            images = pipeline(
                generator=generator,
                lq_image=LR_image_init,
                strength=strength,
                num_inference_steps=args.num_inference_steps,
                num_images_per_lq=args.eval_batch_size,
                output_type="numpy",
            ).images
        except Exception as e:
            print(e)
            continue

        images_processed = (images * 255).round().astype("uint8")
        for i in range(args.eval_batch_size):
            cv2.imwrite(f'{args.output_dir}/LR/{filename}_{i:02d}.png', images_processed[i,:,:,:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default="runs/ddpm_did/checkpoint-50000")
    parser.add_argument("--hq_folder", type=str, default="datasets/DIV2K_train_HR_sub")
    parser.add_argument("--output_dir", type=str, default="datasets/HRLR_Realistic")
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_strength", type=float, default=0.25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
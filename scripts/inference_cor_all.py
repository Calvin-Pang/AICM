import torch
import argparse
import logging
from tensorboardX import SummaryWriter
import os
import numpy as np
from cm.MRI_datasets import *
from torch.utils.data.dataloader import DataLoader
import random, itertools
from PIL import Image
from torch.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import glob
from pytorch_msssim import ssim
from lpips import LPIPS
from datetime import datetime
from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_sam_encoder
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from cm.utils import *
import argparse
import sys
import warnings
import datetime
warnings.filterwarnings("ignore")
# torch.manual_seed(42)
# random.seed(0)
# np.random.seed(0)
torch.cuda.set_device(3)

def compute_psnr(sr, hr, data_range = 1.0):
    diff = (sr - hr) / data_range
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse + 1e-10)

def compute_ssim(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range)


def main():
    # parse configs
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir = args.save_dir)    
    logger.log("Command used: " + " ".join(sys.argv))  
    # Initialize WandbLogger
    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    loss_fn_lpips = LPIPS(net='vgg').cuda()

    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    if args.fm:
        logger.log('build SAM encoder')
        sam_encoder = create_sam_encoder()
        sam_encoder.to(dist_util.dev())
        sam_encoder.eval()
        for p in sam_encoder.parameters():
            p.requires_grad = False
        ckpt = torch.load(args.fm, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        def remap(k):
            for pre in ('module.image_encoder.', 'image_encoder.', 'module.'):
                if k.startswith(pre):
                    return k[len(pre):]
            return k
        new_state = { remap(k): v for k, v in state.items() if 'image_encoder' in k or 'module.image_encoder' in k }
        sam_encoder.load_state_dict(new_state, strict=True)
        diffusion.sam_encoder = sam_encoder
        logger.log('load SAM encoder from', args.fm)

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None
    
    root_path = '/raid/kaifengpang/datasets/IDX_cases_train_test/test'
    files = files = glob.glob(os.path.join(root_path, '**', f'*cor*.npy'), recursive=True)
    num_files = len(files)
    logger.info('Total volumes:', num_files)
    save_dir = os.path.join(logger.get_dir(), 'test_imgs')
    os.makedirs(save_dir, exist_ok = True)

    test_loader = load_data(root_path = root_path, 
                            batch_size = 1,
                            scan = 'cor', 
                            scale = 5.76, 
                            image_size = 320, 
                            crop = 288, 
                            select_k = None,
                            ds = 'sp',
                            mode = 'test',
                            input_mode = args.input_mode)

    psnr_all = []
    ssim_all = []
    lpips_all = []
    logger.info('Begin Model Evaluation.')
    with torch.no_grad():
        for v in tqdm(test_loader, desc = 'Inference...'):
            hr_img = v['hr_img'].to(dist_util.dev())
            h, w = hr_img.shape[-2 : ]
            hr_inte = v['hr_inte'].to(dist_util.dev())
            sam_features = None

            # time_s = datetime.datetime.now()
            if diffusion.sam_encoder is not None:
                h, w = hr_inte.shape[-2], hr_inte.shape[-1]
                img_3c = hr_inte.repeat(1, 3, 1, 1) if args.input_mode == 'single' else hr_inte[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                img_3c = (img_3c + 1) / 2
                # repeat to 3 channel, and pad to max(h,w), max(h,w)
                if h > w:
                    padded_size = h
                    img_3c = F.pad(img_3c, (0, h - w, 0, 0), mode="constant", value=0)
                elif w > h:
                    padded_size = w
                    img_3c = F.pad(img_3c, (0, 0, 0, w - h), mode="constant", value=0)
                else:
                    padded_size = h
                # resize to 1024 * 1024
                img_3c = F.interpolate(img_3c, size = (1024, 1024), mode = "bicubic", align_corners = False)
                img_3c = img_3c.to(next(diffusion.sam_encoder.parameters()).device)
                with torch.no_grad():
                    sam_feature_list = diffusion.sam_encoder(img_3c)
                # resize back
                sam_features = []
                for idx in range(len(sam_feature_list)):
                    sam_features.append(F.interpolate(sam_feature_list[idx].detach(), size = (padded_size, padded_size), mode = "bilinear", align_corners = False)[:, :, :h, :w])

            sample = karras_sample(
                    diffusion,
                    model,
                    (1,1,h,w),
                    steps=1281,
                    hr_inte = hr_inte, 
                    sam_features = sam_features,
                    model_kwargs={},
                    device=dist_util.dev(),
                    clip_denoised=True,
                    sampler='onestep',
                    generator=None,
                    ts=None,
                )
            # time_e = datetime.datetime.now()
            # logger.info(f'CM Inference time: {(time_e - time_s).total_seconds():.6f} seconds.')

            sample = ((sample + 1) / 2).clamp(0, 1) 
            sample = sample.contiguous()
            hr_img = ((hr_img + 1) / 2).clamp(0, 1)
            hr_inte = ((hr_inte + 1) / 2).clamp(0, 1)

            # evaluate psnr, ssim and lpips
            psnr_all.append(compute_psnr(sample, hr_img, data_range=1.0).item())
            ssim_all.append(compute_ssim(sample, hr_img, data_range=1.0).item())
            lpips_all.append(loss_fn_lpips(sample, hr_img).item())

            sample = Image.fromarray((sample.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_img = Image.fromarray((hr_img.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_inte = Image.fromarray((hr_inte[:, 1, :, :].squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L') if args.input_mode == 'stack' else Image.fromarray((hr_inte.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            combined_img = Image.new('L', (w * 3 + 20, h))
            combined_img.paste(hr_inte, (0, 0))
            combined_img.paste(hr_img, (w + 10, 0))
            combined_img.paste(sample, (w * 2 + 20, 0))
            combined_img.save(os.path.join(save_dir, v['img_name'][0]))

    logger.info(f'Average PSNR: {np.mean(psnr_all):.6f} dB')
    logger.info(f'Average SSIM: {np.mean(ssim_all):.6f}')
    logger.info(f'Average LPIPS: {np.mean(lpips_all):.6f}')
            

def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=1281,
        model_path="",
        seed=42,
        ts="",
        save_dir="",
        scan = 'tra',
        crop = 288,
        scale = 5.76,
        channel_mult = (1, 2, 4, 8, 16),
        fm = None, # path to SAM encoder ckpt
        input_mode = 'stack',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
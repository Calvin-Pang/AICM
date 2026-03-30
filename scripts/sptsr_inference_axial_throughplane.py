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
from sptsr.model import *
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
warnings.filterwarnings("ignore")
# torch.manual_seed(42)
# random.seed(0)
# np.random.seed(0)
torch.cuda.set_device(4)


def richardson_lucy_batched(y, psf, num_iter=50, eps=1e-8):

    y = y.cuda()

    B, L = y.shape
    K = psf.shape[0]
    y = y.unsqueeze(1) 
    y = F.pad(y, (K // 2, K // 2), mode='replicate')
    x = torch.ones_like(y).to(y.device)

    psf = psf.reshape(1, 1, -1).to(y.device)

    # pad = K // 2
    

    for _ in tqdm(range(num_iter), desc = 'Deconvolving', leave = False):
        conv_x = F.conv1d(x, psf, padding='same')
        ratio = y / (conv_x + eps)
        correction = F.conv1d(ratio, psf,padding='same')
        x = x * correction

    x = x[:, :, K // 2 : - (K // 2)]
    return x.squeeze(1)

def main():
    # parse configs

    save_root = 'save_sptsr/test_sptsr_axi_throughplane_epoch40'
    if not os.path.exists(save_root): os.mkdir(save_root)
    val_root = os.path.join(save_root, 'test_imgs')
    if not os.path.exists(val_root): os.mkdir(val_root)


    netG = Generator()
    ckpt_path = '/raid/kaifengpang/iCM_MRI_throughplane/save_sptsr/train_sptsr_new/checkpoints/netG_40.pth'
    netG.load_state_dict(torch.load(ckpt_path))
    netG.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    print('load generator from', ckpt_path)



    sp = np.abs(scipy.io.loadmat('/raid/kaifengpang/SPTSR/SPTSR_data_prep/prostate_sl_profile_ds_abs.mat')['sl_profile_ds_abs'])
    sp_norm = sp / sp.sum()
    scale = 5.76
    crop = 288
    root_path = '/raid/kaifengpang/datasets/IDX_cases_train_test/test'
    files = glob.glob(os.path.join(root_path, '**', f'*tra*.npy'), recursive=True)
    num_files = len(files)
    print('Total volumes:', num_files)

    print('Begin Model Evaluation.')
    with torch.no_grad():
        for file_i, f in enumerate(files):
            print(f"Processing file {file_i + 1}/{num_files}: {f}")

            volume_name = f.split('/')[-1].split('.')[0]
            os.makedirs(os.path.join(save_root, 'test_imgs', volume_name), exist_ok = True)
            volume = torch.tensor(np.load(f)).float() # n, h, w
            volume = (volume - volume.min()) / (volume.max() - volume.min())

            # axial SP conv
            if crop:
                crop_h = (volume.shape[1] - crop) // 2
                volume = volume[ : , crop_h : volume.shape[1] - crop_h, : ]
                volume_conv = torch.zeros(volume.shape[0], crop, volume.shape[2])
                for i in tqdm(range(volume.shape[0]), desc = 'Applying SP to axial', leave = False):
                    volume_conv[i] = resize_SP(volume[i].unsqueeze(0), (int(crop // scale), volume.shape[2]), smooth = True) 

            volume_conv = volume_conv.permute(1, 0, 2) # h, n, w
            lr_h, lr_w = volume_conv.shape[-2:] # n, w
            hr_h, hr_w = lr_h * scale, lr_w
            if hr_h % 16 != 0:
                pad = 0
                while ((lr_h + pad) * scale) % 16 != 0:
                    pad += 1
                volume_conv_pad = F.pad(volume_conv, (0, 0, 0, pad), mode="reflect", value=0)
                target_h = int((lr_h + pad) * scale)

            volume_sr = torch.zeros(volume_conv_pad.shape[0], target_h, hr_w)
            volume_lr = torch.zeros(volume_conv_pad.shape[0], target_h, hr_w)
            img_i = 0
            for lr in tqdm(volume_conv_pad, leave = False):
                img_i += 1

                lr = lr.unsqueeze(0).flip(-2)# 1, n, w  20 * 320                
                lr_prev = volume_conv_pad[img_i - 2].unsqueeze(0).flip(-2) if img_i > 1 else lr
                lr_next = volume_conv_pad[img_i].unsqueeze(0).flip(-2) if img_i < volume_conv_pad.shape[0] - 1 else lr

                hr_inte = resize_bicubic(lr, (target_h, hr_w)).unsqueeze(0).float() # 1, n*scale, w

                lr_stack = torch.cat([lr_prev, lr, lr_next], dim=0) # 1, 3*n, w



                h, w = hr_inte.shape[-2 : ]
                lr_stack = lr_stack.to(dist_util.dev())
                hr_inte = hr_inte.to(dist_util.dev())

                sample = y_fake = netG(lr_stack.unsqueeze(0)).clamp(0, 1)
                sample  = nn.functional.interpolate(y_fake, size=(target_h, hr_w), mode='bicubic', align_corners=False)

                sr_img = sample.contiguous()
                lr_img = hr_inte.clamp(0, 1)

                volume_sr[img_i - 1] = sr_img[0]
                volume_lr[img_i - 1] = lr_img[0]

            volume_sr = volume_sr[:,  -int(scale * lr_h) : , :]if hr_h % 16 != 0 else volume_sr # 50, 115, 320
            volume_lr = volume_lr[:,  -int(scale * lr_h) : , :] if hr_h % 16 != 0 else volume_lr # 50, 115, 320


            B, H, W = volume_sr.shape
            volume_sr_flat = volume_sr.permute(1, 2, 0).reshape(-1, volume_sr.shape[0]) 
            volume_sr_deconv_flat = richardson_lucy_batched(volume_sr_flat, torch.tensor(sp_norm[:, 0]).float(), num_iter=500)
            volume_sr_deconv = volume_sr_deconv_flat.view(H, W, B).permute(2, 0, 1)

            volume_lr_flat = volume_lr.permute(1, 2, 0).reshape(-1, volume_lr.shape[0])
            volume_lr_deconv_flat = richardson_lucy_batched(volume_lr_flat, torch.tensor(sp_norm[:, 0]).float(), num_iter=500)
            volume_lr_deconv = volume_lr_deconv_flat.view(H, W, B).permute(2, 0, 1)



            z = volume_sr_deconv.shape[0]
            while z % 5 != 0:
                z -= 1
            volume_sr_deconv = volume_sr_deconv[: z]
            volume_lr_deconv = volume_lr_deconv[: z]
            volume_sr = volume_sr[: z]

            volume_sr = volume_sr.view(z // 5, 5, volume_sr.shape[1], volume_sr.shape[2])
            volume_sr = volume_sr.mean(dim=1)
            volume_sr_deconv_mean = volume_sr_deconv.view(z // 5, 5, volume_sr_deconv.shape[1], volume_sr_deconv.shape[2])
            volume_sr_deconv_mean = volume_sr_deconv_mean.mean(dim=1) 
            volume_lr_deconv_mean = volume_lr_deconv.view(z // 5, 5, volume_lr_deconv.shape[1], volume_lr_deconv.shape[2])
            volume_lr_deconv_mean = volume_lr_deconv_mean.mean(dim=1)

            for i in range(volume_sr_deconv_mean.shape[0]):
                img_name = volume_name + '_TP_' + str(i * 5 + i)
                transforms.ToPILImage()(volume_sr[i]).convert('L').save('{}/{}/{}/{}_sr_conv.png'.format(save_root, 'test_imgs', volume_name, img_name))
                transforms.ToPILImage()(volume_lr_deconv_mean[i]).convert('L').save('{}/{}/{}/{}_lr.png'.format(save_root, 'test_imgs', volume_name, img_name))
                transforms.ToPILImage()(volume_sr_deconv_mean[i]).convert('L').save('{}/{}/{}/{}_sr_deconv.png'.format(save_root, 'test_imgs', volume_name, img_name))


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
        fm = None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
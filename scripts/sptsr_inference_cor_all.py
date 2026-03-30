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
torch.cuda.set_device(3)

def compute_psnr(sr, hr, data_range = 1.0):
    diff = (sr - hr) / data_range
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse + 1e-10)

def compute_ssim(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range)


def main():
    # parse configs
    save_root = 'debug_1119'#'save_sptsr/test_sptsr_cor_epoch60'
    if not os.path.exists(save_root): os.mkdir(save_root)
    val_root = os.path.join(save_root, 'test_imgs')
    if not os.path.exists(val_root): os.mkdir(val_root)
    log_file = os.path.join(save_root, 'test_log.txt')
    if os.path.exists(log_file): os.remove(log_file)

    loss_fn_lpips = LPIPS(net='vgg').cuda()

    test_dataset = MRIImageFolder_SPTSR(root_path = '/raid/kaifengpang/datasets/IDX_cases_train_test/test',
                                    scan = 'cor',
                                    scale = 5.76,
                                    image_size = 320,
                                    crop = 288,
                                    select_k = None,
                                    ds = 'sp')
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers = 8)


    netG = Generator()
    ckpt_path = '/raid/kaifengpang/iCM_MRI_throughplane/save_sptsr/train_sptsr_new/checkpoints/netG_40.pth'
    netG.load_state_dict(torch.load(ckpt_path))
    netG.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)

    psnr_all = []
    ssim_all = []
    lpips_all = []
    print('Begin Model Evaluation.')
    with open(log_file, 'a') as f:
        f.write(f'Begin Model Evaluation.')
    with torch.no_grad():
        for v in tqdm(test_loader, desc = 'Inference...'):
            lr = ((v['lr_img'] + 1) / 2).to(device)
            hr = ((v['hr_img'] + 1) / 2).to(device)
            hr_inte = ((v['hr_inte'] + 1) / 2).to(device)
            h, w = hr.shape[-2 : ]
            time_s = datetime.now()
            y_fake = netG(lr).clamp(0, 1)
            sample = nn.functional.interpolate(y_fake, size=(288, 320), mode='bicubic', align_corners=False)
            time_e = datetime.now()
            print(f'Inference time: {(time_e - time_s).total_seconds():.6f} seconds.')
            hr_img = hr.clamp(0, 1)
            hr_inte = hr_inte.clamp(0, 1)

            # evaluate psnr, ssim and lpips
            psnr_all.append(compute_psnr(sample, hr_img, data_range=1.0).item())
            ssim_all.append(compute_ssim(sample, hr_img, data_range=1.0).item())
            lpips_all.append(loss_fn_lpips(sample, hr_img).item())

            sample = Image.fromarray((sample.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_img = Image.fromarray((hr_img.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_inte = Image.fromarray((hr_inte.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            combined_img = Image.new('L', (w * 3 + 20, h))
            combined_img.paste(hr_inte, (0, 0))
            combined_img.paste(hr_img, (w + 10, 0))
            combined_img.paste(sample, (w * 2 + 20, 0))
            combined_img.save(os.path.join(val_root, v['img_name'][0]))

    print(f'Average PSNR: {np.mean(psnr_all):.6f} dB')
    print(f'Average SSIM: {np.mean(ssim_all):.6f}')
    print(f'Average LPIPS: {np.mean(lpips_all):.6f}')

    with open(log_file, 'a') as f:
        f.write(f'\nAverage PSNR: {np.mean(psnr_all):.6f} dB')
        f.write(f'\nAverage SSIM: {np.mean(ssim_all):.6f}')
        f.write(f'\nAverage LPIPS: {np.mean(lpips_all):.6f}')
            


if __name__ == "__main__":
    main()
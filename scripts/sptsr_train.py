import torch
import os
import numpy as np
from cm.MRI_datasets import *
from torch.utils.data.dataloader import DataLoader
import random
from PIL import Image
import numpy as np
from cm.utils import *
import warnings
import torch.nn as nn
import torch.optim as optim
from sptsr.model import *
from sptsr.loss import *
import time
from torch.optim.lr_scheduler import MultiStepLR
warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(0)
np.random.seed(0)
torch.cuda.set_device(9)

class Lr_scheduler():
    def __init__(self, batch_size, dataset_len, num_epochs, lr_init, lr_final):
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.num_epochs = num_epochs
        self.lr_init = lr_init
        self.lr_final = lr_final

    # batch is 0-based, epoch is 1-based
    def get_lr(self, batch_idx, epoch_idx):
        if epoch_idx > self.num_epochs or self.lr_init == self.lr_final:
            return self.lr_final

        total_steps = self.num_epochs * self.dataset_len / self.batch_size
        current_step = ((epoch_idx-1) * self.dataset_len / self.batch_size) + batch_idx
        lr_multiplier = (self.lr_final / self.lr_init) ** (current_step / total_steps)
        return self.lr_init * lr_multiplier
    
save_root = 'save_sptsr/train_sptsr_new'
if not os.path.exists(save_root): os.mkdir(save_root)
ckpt_root = os.path.join(save_root, 'checkpoints')
if not os.path.exists(ckpt_root): os.mkdir(ckpt_root)
val_root = os.path.join(save_root, 'val_imgs')
if not os.path.exists(val_root): os.mkdir(val_root)

# create a txt log
log_file = os.path.join(save_root, 'training_log.txt')
if os.path.exists(log_file): os.remove(log_file)

# Hyperparameters
batch_size = 16
num_epochs = 1000
learning_rate = 1e-4
log_interval = 50
G_update_interval = 1


train_datase = MRIImageFolder_SPTSR(root_path = '/raid/kaifengpang/datasets/IDX_cases_train_test/train',
                                    scan = 'cor',
                                    scale = 5.76,
                                    image_size = 320,
                                    crop = 288,
                                    select_k = None,
                                    ds = 'sp')
train_loader = DataLoader(dataset = train_datase, batch_size = batch_size, shuffle = True, num_workers = 8)

    
val_dataset = MRIImageFolder_SPTSR(root_path = '/raid/kaifengpang/datasets/IDX_cases_train_test/test',
                                   scan = 'cor',
                                   scale = 5.76,
                                   image_size = 320,
                                   crop = 288,
                                   select_k = 2,
                                   ds = 'sp')
val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False, num_workers = 8)


# Initialize model, loss function, and optimizer
netG = Generator()
netD = Discriminator_WGAN_gp()
G_loss = GeneratorLoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate)

lr_scheduler = Lr_scheduler(batch_size = batch_size,
                            dataset_len = len(train_loader.dataset),
                            num_epochs = 2,
                            lr_init = learning_rate,
                            lr_final = learning_rate)

# dataset len
print(f'Training dataset size: {len(train_loader.dataset)}')
with open(log_file, 'a') as f:
    f.write(f'Training dataset size: {len(train_loader.dataset)}\n')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)
G_loss.to(device)

# Training loop
for epoch in range(num_epochs):
    # ------- Training -------
    start_time = time.time()
    print(f'Starting training epoch {epoch + 1}\n')
    # print log
    with open(log_file, 'a') as f:
        f.write(f'Starting training epoch {epoch + 1}\n')

    netG.train()
    netD.train()

    running_g_loss = 0.0
    running_adv_loss = 0.0
    running_mse_loss = 0.0
    running_l1_loss = 0.0
    running_perception_loss = 0.0
    running_tv_loss = 0.0
    running_sc_loss = 0.0
    running_d_loss = 0.0
    running_gradient_penalty = 0.0

    for i, data in enumerate(train_loader):
        # Move data to the correct device
        # lr, hr, hr_inte = data['lr_img'].to(device), data['hr_img'].to(device), data['hr_inte'].to(device)
        
        lr = ((data['lr_img'] + 1) / 2).to(device)
        hr = ((data['hr_img'] + 1) / 2).to(device)
        hr_inte = ((data['hr_inte'] + 1) / 2).to(device)

        y_fake = netG(lr)
        # resize to 288*320
        y_fake = nn.functional.interpolate(y_fake, size=(288, 320), mode='bicubic', align_corners=False)

        netD.zero_grad()
        D_out_real = netD(hr).mean()
        D_out_fake = netD(y_fake).mean()

        gradient_penalty = Compute_gradient_penalty(
            netD, hr, y_fake, gp_mode = '1-GP') * 10

        d_loss = - D_out_real + D_out_fake + gradient_penalty
        d_loss.backward(retain_graph=True)
        for g in optimizerD.param_groups:
            g['lr'] = lr_scheduler.get_lr(i, epoch + 1)
        optimizerD.step()

        if i % G_update_interval == G_update_interval - 1:
            netG.zero_grad()
            D_out_fake = netD(y_fake).mean()
            [g_loss, adv_loss, mse_loss, l1_loss, perception_loss,
                tv_loss, self_consistency_loss] = G_loss(D_out_fake, lr, y_fake, hr, [1e-3, 1, 0, 6e-3, 0, 0])
            g_loss.backward()
            for g in optimizerG.param_groups:
                g['lr'] = lr_scheduler.get_lr(i, epoch + 1)
            optimizerG.step()

            running_g_loss += g_loss.item()
            running_adv_loss += adv_loss.item()
            running_mse_loss += mse_loss.item()
            running_perception_loss += perception_loss.item()
        
        running_d_loss += d_loss.item()

        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step: [{i + 1}/{len(train_loader)}], G Loss: {running_g_loss/(log_interval // G_update_interval):.5f}, D Loss: {running_d_loss/(log_interval // G_update_interval):.5f}, adv loss: {running_adv_loss/(log_interval // G_update_interval):.5f}, MSE Loss: {running_mse_loss/(log_interval // G_update_interval):.5f}, Perception Loss: {running_perception_loss/(log_interval // G_update_interval):.5f}, Time: {time.time() - start_time:.2f} sec')
            with open(log_file, 'a') as f:
                f.write(f'Epoch [{epoch+1}/{num_epochs}], G Loss: {running_g_loss/(log_interval // G_update_interval):.5f}, D Loss: {running_d_loss/(log_interval // G_update_interval):.5f}, adv loss: {running_adv_loss/(log_interval // G_update_interval):.5f}, MSE Loss: {running_mse_loss/(log_interval // G_update_interval):.5f}, Perception Loss: {running_perception_loss/(log_interval // G_update_interval):.5f}, Time: {time.time() - start_time:.2f} sec\n')

    # Optional: Save the model checkpoint
    if (epoch + 1) % 20 == 0:
        torch.save(netG.state_dict(),os.path.join(ckpt_root, 'netG_' + str(epoch+1) + '.pth'))
    if (epoch + 1) % 1 == 0:   
        netG.eval() 
        val_dir_now = os.path.join(val_root, 'epoch_' + str(epoch+1))
        if not os.path.exists(val_dir_now): os.mkdir(val_dir_now)
        for i, data in enumerate(val_loader):
            lr = ((data['lr_img'] + 1) / 2).to(device)
            hr = ((data['hr_img'] + 1) / 2).to(device)
            hr_inte = ((data['hr_inte'] + 1) / 2).to(device)
            h, w = hr.shape[-2 : ]
            y_fake = netG(lr).clamp(0, 1)
            outputs = nn.functional.interpolate(y_fake, size=(288, 320), mode='bicubic', align_corners=False)
            
            sr_img = Image.fromarray((outputs.detach().squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_img = Image.fromarray((hr.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            hr_inte = Image.fromarray((hr_inte.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode = 'L')
            combined_img = Image.new('L', (w * 3 + 20, h))
            combined_img.paste(hr_inte, (0, 0))
            combined_img.paste(hr_img, (w + 10, 0))
            combined_img.paste(sr_img, (w * 2 + 20, 0))
            combined_img.save('{}/{}_{}.png'.format(val_dir_now, epoch + 1, i + 1))
        netG.train()

# Final model save
torch.save(netG.state_dict(), os.path.join(save_root, "sptsr_ckpt_final.pth"))
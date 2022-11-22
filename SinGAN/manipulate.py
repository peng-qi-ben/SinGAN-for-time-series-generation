from __future__ import print_function

import pandas as pd

import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments


def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s=None, scale_v=1, scale_h=1, n=0, gen_start_scale=0,
                    num_samples=1000):
    # if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(padding=(0, 0, int(pad1), int(pad1)))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        if type(nzx) == int:
            pass
        else:
            nzx = int(nzx)
        nzy = 1

        images_prev = images_cur
        images_cur = []
        ts_to_save = pd.DataFrame([])
        for i in range(0, num_samples, 1):
            if n == 0:
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, 1, z_curr.shape[2], 1)
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                # I_prev = m(I_prev)
                # I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                # I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                # if G_z.shape[2] < real_next.shape[2]:
                #     G_z = functions.upsampling(G_z, real_next.shape[2], 1)
                I_prev = functions.tsresize(I_prev, 1 / opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), :]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:, :, 0:z_curr.shape[2], :]
                    I_prev = functions.upsampling(I_prev, z_curr.shape[2], 1)
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt
            z_curr = z_curr.to(opt.device)
            I_prev = I_prev.to(opt.device)
            z_in = noise_amp * (z_curr) + I_prev
            I_curr = G(z_in.detach(), I_prev)

            if n == len(reals) - 1:
                if opt.mode in ['train','random_samples']:
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (
                        opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                ts_to_save_temp = pd.Series(np.squeeze(I_curr.cpu().detach()))
                ts_to_save[i] = ts_to_save_temp
                # ts_to_save.to_excel('%s/%d.xlsx' % (dir2save, i))
            images_cur.append(I_curr)
        n += 1
    # Output the sample
    if opt.use_ret == 1:
        path = '%s/min=%d,max=%d,epoch=%d,factor=%f,scale=%d' % (
            dir2save, opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init, opt.scale_ret)
        try:
            os.makedirs(path)
        except:
            pass
        print('Export yield data')
        ts_to_save = ts_to_save / opt.scale_ret
        # Inverted, in order to facilitate the testing of 6+2 indicators
        ts_to_save = ts_to_save.T
        ts_to_save.to_excel('%s/min=%d,max=%d,epoch=%d,factor=%f,scale=%d/result_r.xlsx' % (
            dir2save, opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init, opt.scale_ret))
    else:
        path = '%s/min=%d,max=%d,epoch=%d,factor=%f' % (
            dir2save, opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init)
        try:
            os.makedirs(path)
        except:
            pass
        print('Export standardized price data')
        ts_to_save = ts_to_save.T
        ts_to_save.to_excel('%s/min=%d,max=%d,epoch=%d,factor=%f/result_pirce_z.xlsx' % (
            dir2save, opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init))
    return I_curr.detach()

import numpy as np
import pandas as pd

import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SinGAN.imresize import imresize


def train(opt, Gs, Zs, reals, NoiseAmp):
    # real_ = functions.read_image(opt)
    real_ = functions.read_excel(opt)
    in_s = 0
    scale_num = 0
    # real = imresize(real_,opt.scale1,opt)
    real = functions.tsresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    nfc_prev = 0
    # scale_cum represents the current level and stop_scale represents the last level (calculated)
    while scale_num < opt.stop_scale + 1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)  # 每四层更新一次（乘以2）
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        # plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        df_tep = pd.Series(np.squeeze(reals[scale_num].cpu()))
        df_tep.to_excel('%s/real_scale.xlsx' % (opt.outf))

        D_curr, G_curr = init_models(opt)
        # If it exists, just load
        if (nfc_prev == opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
        # Training for this layer
        z_curr, in_s, G_curr, df_plot = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)
        # Save each layer's loss and image
        df_plot.plot()
        plt.savefig('%s/%dloss.png' % (opt.outf, scale_num ))
        df_plot.to_excel('%s/%dloss.xlsx' % (opt.outf, scale_num))
        # Reset grads
        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)
        # Store path
        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    # Take out the real series (training sample)
    real = reals[len(Gs)]
    opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
    # opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = 1
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    # padding
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0
    # padding
    m_noise = nn.ZeroPad2d(padding=(0, 0, pad_noise, pad_noise))
    m_image = nn.ZeroPad2d(padding=(0, 0, pad_image, pad_image))

    alpha = opt.alpha
    # fix noise
    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    # Store err and result
    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            # z_opt is the noise after padding
            z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1, 1, opt.nzx, opt.nzy))
            # generate noise
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1, 1, opt.nzx, opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        # Update the parameters of G
        for j in range(opt.Gsteps):

            # train with fake
            # First update of G parameters
            if (j == 0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    # Initialization
                    prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)  
                    z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1

                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev


                else:
                    # If it's not the first update, use draw_concat to update the parameters, and prev to be the upsampled version of the generated image
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    prev = m_image(prev)
                    # same for z_prev 
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                    # loss function
                    criterion = nn.MSELoss()
                    real = real.to(opt.device)
                    z_prev = z_prev.to(opt.device)
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev, centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise_ = noise_.to(opt.device)
                prev = prev.to(opt.device)
                noise = opt.noise_amp * noise_ + prev


            noise = noise.to(opt.device)
            fake = netG(noise.detach(), prev)
            # fake = netG(noise.detach())
            #######
            netG.zero_grad()
            output = netD(fake)
            # D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                loss = nn.MSELoss()
                # if opt.mode == 'paint_train':
                #     z_prev = functions.quant2centers(z_prev, centers)
                #     plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                z_opt = z_opt.to(opt.device)
                z_prev = z_prev.to(opt.device)

                Z_opt = opt.noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output = netD(real).to(opt.device)
            # D_real_map = output.detach()
            errD_real = -output.mean()  # -a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            #######
            noise = noise.to(opt.device)
            fake = netG(noise.detach(), prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.cpu().detach().numpy().item())

        errG2plot.append(errG.cpu().detach().numpy().item() )
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss.cpu().numpy().item())

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            print('errG : %e'%(errG.detach() + rec_loss))
            print('errD : %e'%(errD.detach()))

        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            ts_fake = pd.Series(np.squeeze(fake.cpu().detach()))
            ts_fake.to_excel('%s/fake_sample.xlsx' % (opt.outf))
            ts_Gz = pd.Series(np.squeeze(netG(Z_opt.detach(), z_prev).cpu().detach()))
            ts_Gz.to_excel('%s/G(z_opt).xlsx' % (opt.outf))

            # plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/G(z_opt).png' % (opt.outf),
            #            functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()

    df_plot = pd.DataFrame({'errD': errD2plot, 'errG': errG2plot, 'rec_loss':z_opt2plot})
    functions.save_networks(netG, netD, z_opt, opt)
    return z_opt, in_s, netG , df_plot


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    # G_z initialized with in_s
    G_z = in_s
    if len(Gs) > 0:  
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)

            if opt.mode == 'animation_train':
                pad_noise = 0

            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, 1],
                                                 device=opt.device)
                    z = z.expand(1, 1, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise(
                        [1, Z_opt.shape[2] - 2 * pad_noise, 1], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                # z_in = noise + G_z
                z = z.to(opt.device)
                G_z = G_z.to(opt.device)
                z_in = noise_amp * z + G_z
                # update G_z
                G_z = G(z_in.detach(), G_z)
                # G_z upsampling
                G_z = functions.tsresize(G_z, 1 / opt.scale_factor, opt)
                if G_z.shape[2] < real_next.shape[2]:
                    G_z = functions.upsampling(G_z, real_next.shape[2], 1)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                Z_opt = Z_opt.to(opt.device)
                G_z = G_z.to(opt.device)
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = functions.tsresize(G_z, 1 / opt.scale_factor, opt)
                if G_z.shape[2] < real_next.shape[2]:
                    G_z = functions.upsampling(G_z, real_next.shape[2], 1)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                # if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z.float()


def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr, G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num + 1], Gs[:scale_num],
                                                      Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

            G_curr = functions.reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def init_models(opt):
    # generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)

    return netD, netG

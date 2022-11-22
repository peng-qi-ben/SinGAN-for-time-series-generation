import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
# from skimage import morphology
# from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans
import pandas as pd
from scipy.interpolate import interp1d


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


#（WGAN-GP）
def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# def read_image(opt):
#     x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
#     x = np2torch(x, opt)
#     x = x[:, 0:3, :, :]
#     return x


# def read_image_dir(dir, opt):
#     x = img.imread('%s' % (dir))
#     x = np2torch(x, opt)
#     x = x[:, 0:3, :, :]
#     return x


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    # x = x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.permute((1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))


def load_trained_pyramid(opt, mode_='train'):
    # dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if (os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs, Zs, reals, NoiseAmp


def generate_in2coarsest(reals, scale_v, scale_h, opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else:  # if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s


def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.use_ret == 1):
        dir2save = 'TrainedModels/%s/min=%d,max=%d,epoch=%d,factor=%f,scale=%d,alpha=%d' % (
            opt.input_name[:-4], opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init, opt.scale_ret, opt.alpha)
    else:
        dir2save = 'TrainedModels/%s/min=%d,max=%d,epoch=%d,factor=%f,alpha=%d' % (
            opt.input_name[:-4], opt.min_size, opt.max_size, opt.niter, opt.scale_factor_init, opt.alpha)
    # elif (opt.mode == 'animation_train'):
    #     dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    # elif (opt.mode == 'paint_train'):
    #     dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (
    #         opt.input_name[:-4], opt.scale_factor_init, opt.paint_start_scale)
    # elif opt.mode == 'random_samples':
    #     dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], opt.gen_start_scale)
    # elif opt.mode == 'random_samples_arbitrary_sizes':
    #     dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (
    #         opt.out, opt.input_name[:-4], opt.scale_v, opt.scale_h)
    # elif opt.mode == 'animation':
    #     dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    # elif opt.mode == 'SR':
    #     dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    # elif opt.mode == 'harmonization':
    #     dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    # elif opt.mode == 'editing':
    #     dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    # elif opt.mode == 'paint2image':
    #     dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    #     if opt.quantization_flag:
    #         dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def calc_init_scale(opt):
    in_scale = math.pow(1 / 2, 1 / 3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale, iter_num



# --------------------------------------------------------------------------------------------------------------
# lodading data
def read_excel(opt):
    x = pd.read_excel('%s/%s' % (opt.input_dir, opt.input_name), index_col=0).astype('float32')
    train_data = x.iloc[:, 0].values
    # If using yield data, the logarithmic yield is taken and multiplied by a specific multiple; if using original prices, the scaled data is used
    if opt.use_ret == 1:
        train_data = np.log(train_data[1:] / train_data[:-1]).astype('float32')
        train_data *= opt.scale_ret
    # z-score
    else:
        train_data = (train_data - np.mean(train_data)) / (np.std(train_data))
    train_data = train_data[:, None, None, None]
    train_data = train_data.transpose(3, 2, 0, 1)
    train_data = torch.from_numpy(train_data)
    if not (opt.not_cuda):
        train_data = move_to_gpu(train_data)
    train_data = train_data.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else train_data.type(torch.FloatTensor)
    return train_data


def adjust_ts_scale(real_, opt):
    # opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil(
        # (math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
        (math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1

    scale2stop = math.ceil(
        # math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
        #          opt.scale_factor_init))
        math.log(min([opt.max_size, real_.shape[2]]) / real_.shape[2],
                 opt.scale_factor_init))

    opt.stop_scale = opt.num_scales - scale2stop
    # opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),
    #                  1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    # set scale1
    opt.scale1 = min(opt.max_size / real_.shape[2],
                     1)
    real = tsresize(real_, opt.scale1, opt)
    # opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    # opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
    # set scale_factor
    opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    # scale2stop = math.ceil(
    #     math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
    #              opt.scale_factor_init))
    scale2stop = math.ceil(
        math.log(min([opt.max_size, real_.shape[2]]) / real_.shape[2],
                 opt.scale_factor_init))
    # set the layers of the GAN
    opt.stop_scale = opt.num_scales - scale2stop
    return real

# sampling
def tsresize(ts, scale, opt):
    ts_1d = np.squeeze(ts)
    ts_1d = ts_1d.cpu()
    x = np.linspace(0, len(ts_1d), num=len(ts_1d), endpoint=True)
    f = interp1d(x, ts_1d, 'cubic')
    x_tep = np.linspace(0, len(ts_1d), num=int(scale * len(ts_1d)), endpoint=True)
    ts = torch.from_numpy(f(x_tep).reshape(1, 1, -1, 1))
    ts = ts.type(torch.FloatTensor)
    if not (opt.not_cuda):
        ts = move_to_gpu(ts)

    return ts


def creat_reals_pyramid(real, reals, opt):
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = tsresize(real, scale, opt)
        reals.append(curr_real)
    return reals


def upsampling(ts, sx, sy):
    ts_1d = np.squeeze(ts.cpu().numpy())
    x = np.linspace(0, int(len(ts_1d)), num=len(ts_1d), endpoint=True)
    f = interp1d(x, ts_1d, 'cubic')
    x_tep = np.linspace(0, int(len(ts_1d)), num=sx, endpoint=True)
    return torch.from_numpy(f(x_tep).reshape(1, 1, -1, 1)).to(torch.float32)


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), 1, device=device)
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise.to(torch.float32)

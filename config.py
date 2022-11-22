
"""

SinGAN parameter setting

"""

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', help='task to be done', default='train')
    # workspace:
    #   Whether to train with gpu, if so set to 0
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=1)

    # load, input, save configurations
    #   Location of the training model 
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    #   Set random seeds
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #  Adjustment according to the type of data being trained
    parser.add_argument('--nc_z', type=int, help='noise # channels', default=1)
    parser.add_argument('--nc_im', type=int, help='image # channels', default=1)
    #  Storage location of the output
    parser.add_argument('--out', help='output folder', default='Output')

    # networks hyper parameters:
    #   The number of kernels in the first layer of the coarsest network
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    #   kernel size
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    #   Number of layers per network (all 5)
    parser.add_argument('--num_layer', type=int, help='number of layers', default=5)
    #   Step length
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=0)  # math.floor(opt.ker_size/2)

    # pyramid parameters:
    #   Ratio of downsampling to upsampling (defaulted to 0.75, but sometimes adjusted according to the length of the original sequence to avoid making too many GANs)
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.5)  # pow(0.5,1/6))
    parser.add_argument('--noise_amp', type=float, help='addative noise cont weight', default=0.1)
    #   Adjust according to the length, as the kernel size is 3, the feeling field is 11, so generally set to 15 ~ 25, so that the coarsest layer can cover the original sequence of 1/2 or 3/4
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=15)
    #   Equivalent to or slightly longer than the original sequence
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=310)

    # optimization hyper parameters:
    #   Number of iterations per layer
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train per scale')
    #   gamma in adam optimizer
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    #   learning rate
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps', type=int, help='Generator inner steps', default=3)
    parser.add_argument('--Dsteps', type=int, help='Discriminator inner steps', default=3)
    #   wgan-gp loss parameters
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)

    # Whether to use yield data.
    parser.add_argument('--use_ret', type=int, help='Whether to use yield data', default=1)
    # scale rate
    parser.add_argument('--scale_ret', type=int, help='scale rate', default=20)

    return parser

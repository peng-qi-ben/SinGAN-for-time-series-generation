"""

Generating sequences with trained model


"""


from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='random_samples')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    opt = parser.parse_args()


    opt.input_dir = 'Input/Excel'  # The folder where the data is
    opt.input_name = 'crb.xlsx'  
    opt.not_cuda = 1
    opt.use_ret = 1  
    opt.scale_ret = 50  
    opt.min_size = 20  
    opt.max_size = 400  
    opt.niter = 2000 
    opt.scale_factor = 0.8 
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    real = functions.read_excel(opt)
    functions.adjust_ts_scale(real, opt)
    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)

       


"""
SinGAN: main fuction
If trained with yield data, yield data is generated
If training with price data, it is automatically transformed into scaled price training and generates scaled price data

Training data is in the Input/Excel folder
The generated data is in a subfolder of Output

"""

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from torch.backends import cudnn


if __name__ == '__main__':
    parser = get_arguments()
    # parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    # parser.add_argument('--input_name', help='input image name', required=True)
    # parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()

    '''
    Parameters that are often tuned can be tuned directly below, while other parameters can be tuned in the config file
    '''
    opt.mode = 'train'
    # The folder where the training data is located
    opt.input_dir = 'Input/Excel'  # The folder where the input data is
    opt.input_name = 'crb.xlsx'  # file name of the data
    opt.not_cuda = 1 # Whether to run with cpu, if cpu, then equal to 1; if gpu, then equal to 0
    opt.use_ret = 1  # Whether to use yield data
    opt.scale_ret = 20  # How many times the yield data is multiplied
    opt.min_size = 15  # Minimum sample length (generally use 15~25)
    opt.max_size = 320  # Longest sample length (slightly longer than the real sequence or equally long)
    opt.niter = 1000  # Number of iterations
    opt.scale_factor = 0.7  # Rate of sampling
    opt.manualSeed = 123

    # Ensure model reproducibility
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # parameters
    opt = functions.post_config(opt)
    # Initialization value
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    # Set the storage location
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        # Read in real data
        real = functions.read_excel(opt)
        # Set parameters (such as how many layers in total, etc.)
        functions.adjust_ts_scale(real, opt)
        # train
        train(opt, Gs, Zs, reals, NoiseAmp)
        # Generate data (1000 items)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, num_samples=1000)

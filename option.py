import argparse
import template
from help_func.CompArea import PictureFormat
import re
import random
parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=10,
                    help='number of threads for data loading') #MD
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs') #MD
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./Dataset/TRAINING',
                    help='dataset directory') #MD
parser.add_argument('--dir_demo', type=str, default='./Dataset/TEST',
                    help='demo image directory') #MD
parser.add_argument('--data_train', type=str, default='tracing_data',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='tracing_data',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-4000/801-808',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='npz',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='1',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1023,
                    help='maximum value of RGB') #MD
parser.add_argument('--yuv_range', type=int, default=1023,
                    help='maximum value of YUV')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')


parser.add_argument('--data_type', type=str, default=PictureFormat.INDEX_DIC[PictureFormat.UNFILTEREDRECON]+
                                                     '+'+PictureFormat.INDEX_DIC[PictureFormat.RECONSTRUCTION],
                    help='get data type') #MD
parser.add_argument('--tu_data', type=str, default='BLOCK',
                    help='tu data path name')
parser.add_argument('--tu_data_type', type=str, default='QP')

# Model specifications
parser.add_argument('--model', default='dncnn',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory') # only test
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Option for Global and Spatial data based CNN (GSCNN)
parser.add_argument('--fc', type=int, default=64,
                    help='default number of filters. (Use in GSCNN)')
parser.add_argument('--sc', type=int, default=0,
                    help='Split Luma and Chroma')
parser.add_argument('--more_noisy', type=int, default=0,
                    help='selects nosiy image from n of patchs')
parser.add_argument('--better_patch', type=int, default=0,
                    help='selects better psnr square patch in test')
parser.add_argument('--image_pin_memory', type=int, default=0,
                    help='image load in memory always')



# Option for DnCNN
parser.add_argument('--dncnn_depth', type=int, default=17,
                    help='depth of dncnn model depth')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate') #MD
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type') #MD
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay') #MD

#New#
parser.add_argument('--lr_warm_up', type=int, default=0,
                    help='use learning rate warmup during integer epoch')
parser.add_argument('--lr_cosine', action='store_true',
                    help='use cosine annealing learning rate')

parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration') #MD
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load') #only resume and only test
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint') #only resume
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true', default=True,
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--exp_dir', type=str, default='', help='set experiment dir')

parser.add_argument('--rs', type=str, default='', help='set random search parameter')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')
args.data_type = args.data_type.split('+')
args.tu_data_type = args.tu_data_type.upper().split('+')

if args.rs:
    r = re.compile('(?P<name>\w*)\s*=\s*\[(?P<start>-*\d*\.*\d*),\s*(?P<end>-*\d*\.*\d*)\]')
    if r.match(args.rs) is None:
        raise Exception('Random search value Not Unknown')
    for name, start, end in r.findall(args.rs):
        if name not in vars(args):
            raise Exception('Unknown Random Search Parameter')
        t = type(vars(args)[name])
        if t == int:
            vars(args)[name] = random.randint(int(start), int(end) + 1)
        elif t == float:
            vars(args)[name] = random.uniform(float(start), float(end))
        else:
            raise NotImplementedError





if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False


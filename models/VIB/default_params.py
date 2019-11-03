def load_parser(parser):
    parser.add_argument('--epochs', type=int, default=300, help='Total number of epochs.')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='../models/ib_vgg', help='checkpoints Path')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold of alpha. For pruning.')
    parser.add_argument('--kl-fac', type=float, default=1e-6, help='Factor for the KL term.')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use. Single GPU only.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--ib-lr', type=float, default=-1, 
                        help='Separate learning rate for information bottleneck params. Set to -1 to follow args.lr.')
    parser.add_argument('--ib-wd', type=float, default=-1,
                        help='Separate weight decay for information bottleneck params.'+
                        ' Set to -1 to follow args.weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum')
    parser.add_argument('--mag', type=float, default=9, help='Initial magnitude for the variances.')
    parser.add_argument('--lr-fac', type=float, default=0.5, help='LR decreasing factor.')
    parser.add_argument('--lr-epoch', type=int, default=30, help='Decrease learning rate every x epochs.')
    parser.add_argument('--tb-path', type=str, default='../tb/ib_vgg', help='Path to store tensorboard data.')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='Whether to use batch norm')
    parser.add_argument('--opt', type=str, default='sgd', help='Optimizer. sgd or adam.')
    parser.add_argument('--val', action='store_true', default=False, help='Whether to only evaluate model.')
    parser.add_argument('--cfg', type=str, default='D0', help='VGG net config.')
    parser.add_argument('--data-set', type=str, default='cifar10', help='Which data set to use.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a model to be resumes (with its optimizer states).')
    parser.add_argument('--resume-vgg-vib', type=str, default='',
                        help='Path to pretrained VGG model (with IB params), ignore IB params.')
    parser.add_argument('--resume-vgg-pt', type=str, default='',
                        help='Path to pretrained VGG model (without IB params).')
    parser.add_argument('--init-var', type=float, default=0.01, help='Variance for initializing IB parameters')
    parser.add_argument('--reg-weight', type=float, default=0)
    parser.add_argument('--ban-crop', default=False, action='store_true',
                        help='Whether to ban random cropping after padding.')
    parser.add_argument('--ban-flip', default=False, action='store_true', help='Whether to ban random flipping.')
    parser.add_argument('--sample-train', default=1, type=int,
                        help='Set to non-zero to sample during training.')
    parser.add_argument('--sample-test', default=0, type=int, help='Set to non-zero to sampling during test.')
    parser.add_argument('--no-ib', default=False, action='store_true', help='Ignore IB operators.')
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--workers', type=int, default=1)
    return parser
import os
import argparse

def add_base_config(parser):
    add_model_config(parser)
    add_training_config(parser)
    add_quantization_config(parser)
    add_evaluation_config(parser)

def add_model_config(parser):
    g = parser.add_argument_group('Model Configuration')
    g.add_argument('--model', type=str, default='ResNet', help='The type of base model')
    
    g.add_argument('--resize_size', default=256, type=int)
    g.add_argument('--crop_size', default=224, type=int)
    
    g.add_argument('--data_root', type=str, default= 'data/')
    g.add_argument('--dataset', type=str, default='cifar10')
    g.add_argument('--datalist_root', type=str, default= 'datalist/')
    g.add_argument('--bit_list', type=int, nargs='+')
    g.add_argument('--hash_activation', type=str, default='tanh', help="")
    g.add_argument("--no_pretrained", action='store_true')
        
def add_training_config(parser):
    g = parser.add_argument_group('Training Configuration')
    g.add_argument('--epochs', type=int, default=150)
    g.add_argument('--batch_size', type=int, default=64)
    g.add_argument('--save_path', type=str, required=True)
    g.add_argument("--external_logger", default=None, type=str)
    g.add_argument("--external_logger_args", default=None, type=str)
    g.add_argument('--optimizer', default='rmsprop', type=str)
    g.add_argument('--momentum', default=0.9, type=float)
    g.add_argument('--lr', default=1e-5, type=float)
    g.add_argument('--weight_decay', default=1e-5, type=float)
    g.add_argument('--sched', default=None, type=str)
    g.add_argument('--sched_milestones', type=int, nargs='+')
    g.add_argument('--sched_decay', default=0.1, type=float)
    g.add_argument("--random_seed", default=99, type=int) #only effective for cifar10

    g.add_argument('--workers', default=2, type=int)
def add_quantization_config(parser):
    g = parser.add_argument_group('Quantization Configuration')
    g.add_argument('--quantization_alpha', type=float, default=0.0, help="")
    g.add_argument('--quantization_type', type=str, default=None, help="")
    
def add_evaluation_config(parser): 
    g = parser.add_argument_group('Evaluation Configuration')
    g.add_argument('--test_every', default=10, type=int, help="Number of training epochs per Testing")
    g.add_argument('--save_hash_code', default=False, action='store_true')

    
def setup_config_dataset(args):
    if args.dataset.startswith("cifar10"):
        args.topK = -1
        args.n_class = 10
    elif args.dataset == "nuswide_21_e1":
        args.topK = 5000
        args.n_class = 21        
    elif args.dataset in ["nuswide_21", "nuswide_21_m"]:
        args.topK = 5000
        args.n_class = 21
    elif args.dataset == "nuswide_81_m":
        args.topK = 5000
        args.n_class = 81
    elif args.dataset == "coco":
        args.topK = 5000
        args.n_class = 80
    elif args.dataset == "imagenet":
        args.topK = 5000
        args.n_class = 100
    elif args.dataset == "mirflickr":
        args.topK = -1
        args.n_class = 38
    elif args.dataset == "voc2012":
        args.topK = -1
        args.n_class = 20
    elif args.dataset == 'cifar2':
        args.topK = -1
        args.n_class = 2
    elif args.dataset == 'cifar4':
        args.topK = -1
        args.n_class = 4
    elif args.dataset == 'ucifar10':
        args.topK = -1
        args.n_class = 10
    else:
        raise Exception(f'Dataset Not Supported: {args.dataset}')

    if args.dataset in ['imagenet']:
        args.data_path = os.path.join(args.data_root, args.dataset)
    elif args.dataset in ["nuswide_21", 'nuswide_21_e1']:
        args.data_path = os.path.join(args.data_root, 'nus_wide')
    elif args.dataset in ["nuswide_21_m", "nuswide_81_m"]:
        args.data_path = os.path.join(args.data_root, 'nus_wide_m')
    elif args.dataset == "coco":
        args.data_path = os.path.join(args.data_root, 'COCO_2014')
    elif args.dataset in ['cifar2', 'cifar4', 'cifar10', 'voc2012']:
        args.data_path = args.data_root
    else:
        raise Exception(f'Dataset Not Supported: {args.dataset}')
        
    if not args.data_path.endswith('/'):
        args.data_path += '/'
        
    args.data_train_set = os.path.join(args.datalist_root, args.dataset, 'train.txt')
    args.data_database_set = os.path.join(args.datalist_root, args.dataset, 'database.txt')
    args.data_test_set = os.path.join(args.datalist_root, args.dataset, 'test.txt')
   
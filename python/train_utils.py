import torch
import torch.optim as optim
import timm
from tqdm import tqdm

from network import *
from metrics import ranking

def create_optimizer(args, params):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise Exception(f'Unsupported optimizer {args.optimizer}!')
        
    if args.sched == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.sched_milestones, args.sched_decay)
    elif args.sched is None:
        scheduler = None
    else:
        raise Exception(f'Unsupported scheduler {args.sched}')
    
    return optimizer, scheduler

def create_model(bit, args):
    if args.model == 'ResNet':
        return ResNet(bit, pretrained=not args.no_pretrained)
    elif args.model == 'AlexNet':
        return AlexNet(bit, pretrained=not args.no_pretrained)
    elif args.model.startswith('resnet') or args.model_type in ['vgg11']:
        return timm.create_model(
                args.model, pretrained=not args.no_pretrained, num_classes=bit)
    else:
        raise Exception(f'Invalid mode type {args.model_type}')
        
def compute_result(dataloader, net, device, desc='Binarizing', return_pre_thresholding=False):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader, desc=desc):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    if return_pre_thresholding:
        return torch.cat(bs), torch.cat(bs).sign(), torch.cat(clses)
    else:
        return torch.cat(bs).sign(), torch.cat(clses)        

@torch.no_grad()
def evaluate(args, net, test_loader, database_loader, device):
    tst_binary, tst_label = compute_result(test_loader, net, device=device, desc='Binarizing Test')
    bd_u, bd_binary, bd_label = compute_result(database_loader, net, 
                                               device=device, return_pre_thresholding=True, desc='Binarizing Database')

    R = [1000, bd_binary.size(0) if args.topK == -1 else args.topK]
    
    allPrec, allmAP = ranking.calculate_all_metrics(
        bd_binary.numpy(), bd_label.numpy(), tst_binary.numpy(), tst_label.numpy(),
        R)
    return allPrec, allmAP, tst_binary, tst_label, bd_u, bd_binary, bd_label

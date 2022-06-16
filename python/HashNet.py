import argparse
import os
import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import sys
from termcolor import colored
from tqdm import tqdm

from external_logger import ExternalLogger
from config import add_base_config, setup_config_dataset
from losses.distributional_quantization_losses import quantization_ot_loss, quantization_swdc_loss, quantization_swd_loss
from train_utils import  *
from data_utils import get_data

torch.multiprocessing.set_sharing_strategy('file_system')


# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

class HashNetLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(args.num_train, bit).float().to(args.device)
        self.Y = torch.zeros(args.num_train, args.n_class).float().to(args.device)

        self.scale = 1

    def forward(self, u, y, ind, args):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = args.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss

def train_val(external_logger, net, args, bit, log_lr_stats=False):
    device = args.device
    train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
    
    args.num_train = num_train
    args.num_test = num_test
    args.num_dataset = num_database

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    prefix = f"HashNet-{args.dataset}-{bit}-{args.quantization_type}"
    prefix = f"{prefix}-{args.quantization_alpha}-{args.random_seed}"
        
    checkpoint_file = os.path.join(args.save_path, f'{prefix}-checkpoint.pt')
    
    net = net.to(device)
    print(net)
    
    optimizer, scheduler = create_optimizer(args, net.parameters())

    criterion = HashNetLoss(args, bit)
    
    if os.path.exists(checkpoint_file):
        print(colored(f'Train model from checkpoint {checkpoint_file}!', 'blue'))
        checkpoint = torch.load(checkpoint_file)
        
        net.load_state_dict(checkpoint['net'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['best_mAP']
        criterion.scale = checkpoint['criterion_scale']
    else:
        print(f'Train model from sratch!')
        start_epoch = 0
        best_mAP = 0

    for epoch in range(start_epoch, args.epochs):
        criterion.scale = (epoch // args.step_continuation + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        

        net.train()

        train_loss = 0
        train_quantization_loss = 0
        pbar = tqdm(train_loader, total=len(train_loader), file=sys.stdout)
        for image, label, ind in pbar:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, args)
            train_loss += loss.item()
            
            if args.quantization_type == 'ot':
                quantization_loss = quantization_ot_loss(u)
            elif args.quantization_type == 'swd':
                quantization_loss = quantization_swd_loss(u.view(u.size(0), -1))
            elif args.quantization_type == 'swdC':
                quantization_loss = quantization_swdc_loss(u.view(u.size(0), -1))
            else:
                quantization_loss = torch.tensor(0.0)
            train_quantization_loss += quantization_loss.item()
            
            (loss + args.quantization_alpha * quantization_loss).backward()          
            optimizer.step()
            pbar.set_description(
                '[BIT{}-{:02d}/{:02d}: {}][scale:{:0.3f},lr:{:.06f})]: l_c {:.04f} l_q {:.04f}'.format(
                bit, epoch+1, args.epochs, current_time, criterion.scale, optimizer.param_groups[0]['lr'], 
                train_loss / len(train_loader), train_quantization_loss / len(train_loader)))
            
        pbar.close()
        
        if external_logger:
            external_logger.log_val(f'{bit}-locality_loss', train_loss / len(train_loader), epoch+1)
            external_logger.log_val(f'{bit}-quantization_loss', train_quantization_loss / len(train_loader), epoch+1)
            external_logger.set_val('e', epoch+1)
            if log_lr_stats:
                external_logger.log_val('train_lr', optimizer.param_groups[0]['lr'], epoch+1)
                                
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % args.test_every == 0 or (epoch + 1) == args.epochs:
            net.eval()
            all_prec, all_mAP, tst_binary, tst_label, bd_u, bd_binary, bd_label = evaluate(
                args, net, test_loader, database_loader, device)
            
            mAP = all_mAP[bd_binary.size(0) if args.topK == -1 else args.topK]
            prec1000 = all_prec[1000]
            mAP1000 = all_mAP[1000]
            
            if external_logger:
                external_logger.log_val(f'{bit}-mAP', mAP, epoch+1)
                external_logger.log_val(f'{bit}-prec1000', prec1000, epoch+1)
                external_logger.log_val(f'{bit}-mAP1000', mAP1000, epoch+1)
            print('[BIT{}-{:02d}/{:02d}: {}] Retrieval mAP {:.04f} prec@1000 {:.04f} mAP@1000 {:.04f}'.format(
                bit, epoch+1, args.epochs, current_time, mAP, prec1000, mAP1000))


            if mAP > best_mAP:
                print(colored('[BIT{}-{:02d}/{:02d}: {}] Testing: Found new best retrieval {:.04f} ==> {:.04f}'.format(
                    bit, epoch+1, args.epochs, current_time, best_mAP, mAP), 'red'))

                best_mAP = mAP
                    
                model_file = os.path.join(args.save_path, f'{prefix}-bestmodel.pt')
                print("Save best model in {}".format(model_file))
                
                torch.save({
                    'args': args, 'net': net.state_dict(), 'best_mAP': best_mAP, 'all_mAP': all_mAP, 'all_prec': all_prec 
                }, model_file)

                if args.save_hash_code:
                    code_file = os.path.join(args.save_path, f'{prefix}-binary.npz')                                   
                    np.savez_compressed(
                        code_file,
                        db_binary=db_binary.numpy(), db_label=db_label.numpy(),
                        tst_binary=tst_binary.numpy(), tst_label=tst_label.numpy(),
                        mAP=all_mAP, prec=all_prec
                    )
        
        torch.save({
           'epoch': epoch, 'args': args, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
           'scheduler': scheduler.state_dict() if scheduler else None,
           'criterion_scale': criterion.scale,
           'best_mAP': best_mAP, 
       }, checkpoint_file)           
      
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HashNet')
    add_base_config(parser)
    
    # Method's specific configurations
    g = parser.add_argument_group('Method Config')
    g.add_argument('--step_continuation', default=20, type=int)
    g.add_argument('--alpha', default=0.1, type=float)
    
    args = parser.parse_args()
    setup_config_dataset(args)
    
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    
    if args.external_logger:
        external_logger = ExternalLogger(args)
        external_logger.set_dict(args.__dict__)
        external_logger.set_val('args', args)
    else:
        external_logger = None
    
    for bit in args.bit_list:
        net = create_model(bit, args)        
        train_val(external_logger, net, args, bit, log_lr_stats=(bit==args.bit_list[0]))

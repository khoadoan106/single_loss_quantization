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


# DSDH(NIPS2017)
# paper [Deep Supervised Discrete Hashing](https://papers.nips.cc/paper/6842-deep-supervised-discrete-hashing.pdf)
# code [DSDH_PyTorch](https://github.com/TreezzZ/DSDH_PyTorch)

class DSDHLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super().__init__()
        self.U = torch.zeros(bit, args.num_train).float().to(args.device)
        self.B = torch.zeros(bit, args.num_train).float().to(args.device)
        self.Y = torch.zeros(args.n_class, args.num_train).float().to(args.device)

    def forward(self, u, y, ind, args):

        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        inner_product = u @ self.U * 0.5
        s = (y @ self.Y > 0).float()

        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        # Classification loss
        cl_loss = (y.t() - self.W.t() @ self.B[:, ind]).pow(2).mean()

        # Regularization loss
        reg_loss = self.W.pow(2).mean()

        loss = likelihood_loss + args.mu * cl_loss + args.nu * reg_loss
        return loss

    def updateBandW(self, args, bit):
        device = args.device
        B = self.B
        for dit in range(args.dcc_iter):
            # W-step
            W = torch.inverse(B @ B.t() + args.nu / args.mu * torch.eye(bit).to(device)) @ B @ self.Y.t()

            for i in range(B.shape[0]):
                P = W @ self.Y + args.eta / args.mu * self.U
                p = P[i, :]
                w = W[i, :]
                W_prime = torch.cat((W[:i, :], W[i + 1:, :]))
                B_prime = torch.cat((B[:i, :], B[i + 1:, :]))
                B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

        self.B = B
        self.W = W

def train_val(external_logger, net, args, bit, log_lr_stats=False):
    device = args.device
    train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(args)
    
    args.num_train = num_train
    args.num_test = num_test
    args.num_dataset = num_database

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    prefix = f"DSDH-{args.dataset}-{bit}-{args.quantization_type}"
    prefix = f"{prefix}-{args.quantization_alpha}-{args.random_seed}"
        
    checkpoint_file = os.path.join(args.save_path, f'{prefix}-checkpoint.pt')
    
    net = net.to(device)
    print(net)
    
    optimizer, scheduler = create_optimizer(args, net.parameters())

    criterion = DSDHLoss(args, bit)
    
    if os.path.exists(checkpoint_file):
        print(colored(f'Train model from checkpoint {checkpoint_file}!', 'blue'))
        checkpoint = torch.load(checkpoint_file)
        
        net.load_state_dict(checkpoint['net'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['best_mAP']
        criterion = checkpoint['criterion']
    else:
        print(f'Train model from sratch!')
        start_epoch = 0
        best_mAP = 0

    for epoch in range(start_epoch, args.epochs):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        criterion.updateBandW(args, bit)

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
                '[BIT{}-{:02d}/{:02d}: {}][lr:{:.06f})]: l_c {:.04f} l_q {:.04f}'.format(
                bit, epoch+1, args.epochs, current_time, optimizer.param_groups[0]['lr'], 
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
           'criterion': criterion,
           'best_mAP': best_mAP, 
       }, checkpoint_file)           
      
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSDH')
    add_base_config(parser)
    
    # Method's specific configurations
    g = parser.add_argument_group('Method Config')
    g.add_argument('--alpha', default=1, type=float)
    g.add_argument('--nu', default=1, type=int)
    g.add_argument('--mu', default=1, type=int)
    g.add_argument('--eta', default=55, type=int)
    g.add_argument('--dcc_iter', default=10, type=int)
    
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

import sys
import os
from pathlib import Path
from argparse import Namespace
import copy

import numpy as np
from tqdm import tqdm
from termcolor import colored
import wandb

class ExternalLogger(object):
    def __init__(self, args, run_name=None):
        if args.external_logger is None:
            self.run = None
            self.logger = None
        elif args.external_logger.startswith('neptune'):
            self.logger = 'neptune'
            project = args.external_logger_args
            if not os.path.exists(os.path.expanduser('~/.neptune')):
                raise Exception('Please create .neptune file to store your credential!')
            api_token = open(os.path.expanduser('~/.neptune')).readline().strip()
            import neptune.new as neptune
            if 'offline' in args.external_logger:
                self.run = neptune.init(
                    project=project,
                     api_token=api_token,
                    name=run_name,
                    mode='offline'
                ) # your credentials
            else:
                self.run = neptune.init(
                    project=project,
                     api_token=api_token,
                    name=run_name,
                ) # your credentials
            self.set_val('CMD', " ".join(sys.argv[:]))
        elif args.external_logger.startswith('wandb'):
            project, entity = args.external_logger_args.split('|')
            wandb.init(project=project, entity=entity, name=run_name)
            self.logger = 'wandb'
            self.config = {}
            
    def log_val(self, key, val, step=None):
        if self.logger == 'neptune':
            self.run[key].log(val, step=step)
        elif self.logger == 'wandb':
            wandb.log({key: val}, step=step)
    def set_val(self, key, val):
        if self.logger == 'neptune':
            self.run[key] = val
        elif self.logger == 'wandb':
            #self.config[key] = val
            #wandb.config = self.config
            #wandb.config[key] = val
            wandb.config.update({key: val}, allow_val_change=True)
            
    def log_img(self, key, img, step=None):
        if self.logger == 'neptune':
            from neptune.new.types import File
            if type(img) == str:
                self.run[key].log(File(img), step=step)
        elif self.logger == 'wandb':
            wandb.log({key: wandb.Image(img)}, step=step)
                
    def set_dict(self, d):
        if self.logger == 'neptune':
            for k, v in d.items():
                self.run[k] = v
        elif self.logger == 'wandb':
            #self.config.update(d)
            #wandb.config = self.config
            #for k, v in d.items():
            #    wandb.config[k] = v
            wandb.config.update(d, allow_val_change=True)

    def log_dict(self, d, step=None):
        if self.logger == 'neptune':
            for k, v in d.items():
                self.run[k].log(v, step=step)
        elif self.logger == 'wandb':
            d = copy.deepcopy(d)
            #d['step'] = step
            wandb.log(d, step=step)
                
    def cleanup(self):
        if self.logger == 'neptune':
            self.run.stop()

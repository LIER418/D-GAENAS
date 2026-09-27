import os
import sys
import glob
import time
import numpy as np
import torch

import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from utils import _data_transforms_cifar10, _data_transforms_cifar100
from utils import count_parameters_in_MB
from utils import create_exp_dir
from utils import AverageMeter
from utils import accuracy

from generator import *
from supernet_model import Network
import json


class Trainer:
    
    def __init__( self, 
        dataset='CIFAR10', batch_size=64, 
        learning_rate=0.025, learning_rate_min=0.001, 
        momentum=0.9, weight_decay=3e-4, 
        report_freq=50, gpu=0, 
        epochs=50, 
        init_channels=16, layers=8, 
        cutout=False, cutout_length=16,
        data_root='../data', download=True,
        drop_path_prob=0.3, seed=2, 
        grad_clip=5, save_prefix='EXP',
        init_masks = [], amp=True):
        
        
        if not torch.cuda.is_available():
          print('no gpu device available')
          sys.exit(1)
        
        np.random.seed( seed )
        torch.cuda.manual_seed( seed )
        torch.cuda.set_device(gpu)
        cudnn.enabled=True
        cudnn.benchmark = True
        print('gpu device = %d' % gpu)
        print('dataset=', dataset, 'batch_size=', batch_size, 'learning_rate=', learning_rate, 'learning_rate_min=',
              learning_rate_min, 'momentum=', momentum, 'weight_decay=', weight_decay, 'report_freq=', report_freq, 
              'gpu=', gpu, 'epochs=', epochs, 'init_channels=', init_channels, 'layers=', layers, 'cutout=', cutout,
              'cutout_length=', cutout_length, 'drop_path_prob=', drop_path_prob, 'seed=', seed, 'grad_clip=', grad_clip )
        
        savedirs = "supernet-logs"

        continue_train   = False
        if os.path.exists(savedirs + '/model.pt'):
          continue_train = True
         
        if not continue_train:
          create_exp_dir(savedirs, scripts_to_save=glob.glob('*.py'))
        
        self.dataset           = dataset
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.learning_rate_min = learning_rate_min
        self.momentum          = momentum
        self.weight_decay      = weight_decay
        self.report_freq       = report_freq
        self.epochs            = epochs
        self.init_channels     = init_channels
        self.layers            = layers
        self.cutout            = cutout
        self.cutout_length     = 16
        self.data_root         = data_root
        self.download          = download
        self.drop_path_prob    = drop_path_prob
        self.grad_clip         = grad_clip
        self.save_prefix       = savedirs
        self.start_epoch       = 0
        self.mask_to_train     = init_masks
        if dataset == 'CIFAR10':
            CIFAR_CLASSES      = 10
        elif dataset == 'CIFAR100':
            CIFAR_CLASSES      = 100
        
        supernet_normal = supernet_generator(node, layer_type)
        supernet_reduce = supernet_generator(node, layer_type)

        self.criterion  = nn.CrossEntropyLoss()
        self.criterion  = self.criterion.cuda()
        self.supernet   = Network(supernet_normal, supernet_reduce, layer_type, init_channels, CIFAR_CLASSES, layers, self.criterion, steps=len(supernet_normal))
        self.supernet   = self.supernet.cuda()
        self.optimizer  = torch.optim.SGD( self.supernet.parameters(), self.learning_rate, momentum = self.momentum, weight_decay = weight_decay)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, float(epochs), eta_min = learning_rate_min )
        
        if dataset == 'CIFAR10':
            train_transform, valid_transform = _data_transforms_cifar10(cutout, cutout_length)
            train_data = dset.CIFAR10(root=data_root, train=True, download=download, transform=train_transform)
            valid_data = dset.CIFAR10(root=data_root, train=False, download=download, transform=valid_transform)
        elif dataset == 'CIFAR100':
            train_transform, valid_transform = _data_transforms_cifar100(cutout, cutout_length)
            train_data = dset.CIFAR100(root=data_root, train=True, download=download, transform=train_transform)
            valid_data = dset.CIFAR100(root=data_root, train=False, download=download, transform=valid_transform)

        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
        
        print("length of training & valid queue:", len(self.train_queue), len(self.valid_queue) )
        print("param size = %fMB"% count_parameters_in_MB(self.supernet) )
        
        if continue_train:
            print('continue train from checkpoint')
            checkpoint       = torch.load(self.save + '/model.pt')
            self.supernet.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler   = checkpoint['scheduler']
        
        self.curt_epoch = self.start_epoch
        self.curt_step  = 0
        
        self.amp        = amp
        self.scaler     = torch.cuda.amp.GradScaler(enabled=amp)

        self.base_net   = zero_supernet_generator(node, layer_type )
        
        
    def propose_nasnet_mask(self, nums=1):
        net_mask = np.array( sampled_nets_generator(self.base_net, nums=1))
        return net_mask
    

    def progressive_nasnet_mask(self, epoch, dynamic_path=[7, 5, 3, 1], nums=1):
        if epoch <= 400:
            net_mask = np.array(progressive_sampled_nets_generator(self.base_net, 1, 7))
        else:
            net_mask = np.array(progressive_sampled_nets_generator(self.base_net, 1, 1))
        return net_mask


    def zero_supernet_generator(self):
        vec_length = len(self.layer_type)
        masked_vec = np.zeros((1, vec_length))[0].tolist()
        disconnected_vec = np.zeros((1, vec_length))[0].tolist()
        supernet = [[] for v in range(self.arch_node)]
        for i in range(self.arch_node):
            for j in range(self.arch_node + 2):
                if j < i + 2:
                    supernet[i].append(masked_vec.copy())
                else:
                    supernet[i].append(disconnected_vec.copy())
        for i in range(len(supernet)):
            for j in range(len(supernet[i])):
                for n in range(len(supernet[i][j])):
                    supernet[i][j][n] = int(supernet[i][j][n])
        return supernet


    def train( self ):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        for step, (input, target) in enumerate(self.train_queue):
            mask = self.propose_nasnet_mask()
            curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )            
            self.supernet.change_masks(curt_normal_mask, curt_reduce_mask)
            
            self.supernet.train()
            n = input.size(0)
            
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            with torch.autocast('cuda', enabled=self.amp):
                logits = self.supernet(input)
                loss   = self.criterion(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % self.report_freq == 0:
                print('train steps: %03d loss: %e top-1: %f top-5: %f'% (step, objs.avg, top1.avg, top5.avg) )
        return top1.avg, objs.avg


    def progressive_train(self, epoch):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        for step, (input, target) in enumerate(self.train_queue):
            mask = self.progressive_nasnet_mask(epoch)
            curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )            
            self.supernet.change_masks(curt_normal_mask, curt_reduce_mask)
            
            self.supernet.train()
            n = input.size(0)
            
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            with torch.autocast('cuda', enabled=self.amp):
                logits = self.supernet(input)
                loss   = self.criterion(logits, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % self.report_freq == 0:
                print('train steps: %03d loss: %e top-1: %f top-5: %f'% (step, objs.avg, top1.avg, top5.avg) )
        return top1.avg, objs.avg


    def load_model(self, path, device):
        print('continue train from checkpoint')
        checkpoint       = torch.load(path, map_location=device)
        self.supernet.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler   = checkpoint['scheduler']


    def infer_masks( self, mask ):
        result = {}
        normal_mask, reduce_mask   = encoding_to_masks( mask )
        print("normal mask:", normal_mask)
        print("reduce mask:", reduce_mask)
        valid_acc, valid_obj       = self.infer( [normal_mask, reduce_mask] )
        result[ json.dumps(mask.tolist() ) ] = valid_acc
        return result
    

    def infer( self, mask ):
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.supernet.change_masks( mask[0], mask[1] )

        self.supernet.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.cuda()
                target = target.cuda(non_blocking=True)

                logits = self.supernet(input)
                loss   = self.criterion(logits, target)

                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.report_freq == 0:
                    print('valid step: %03d loss: %e top-1: %f top-5: %f'% ( step, objs.avg, top1.avg, top5.avg ) )

        return top1.avg, objs.avg            


    def run(self):
        best_acc = 0.0
        
        mask = self.propose_nasnet_mask()
        curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )
        print("proposed norm mask:",  curt_normal_mask)   
        print("proposed reduce mask:", curt_reduce_mask)            
                 
        valid_acc, valid_obj = self.infer( [curt_normal_mask, curt_reduce_mask] )
        
        
        train_start = time.time()
        for epoch in range(0, self.epochs):
          epoch_start = time.time()
          print('epoch :', epoch + 1)
          train_acc, train_obj = self.train()

          self.scheduler.step()
          epoch_sec = time.time() - epoch_start
          print('train_acc %f'% train_acc)
          print('epoch time: %.1f s  (%.4f GPU-days so far)' % (
              epoch_sec,
              (time.time() - train_start) / 86400.0,
          ))

          curt_normal_mask, curt_reduce_mask = encoding_to_masks( mask )
          print("proposed norm mask:",  curt_normal_mask)
          print("proposed reduce mask:", curt_reduce_mask)
          valid_acc, valid_obj = self.infer( [curt_normal_mask, curt_reduce_mask] )
          print('valid_acc %f'% valid_acc)
          print('saving the latest model')
          torch.save({'epoch': epoch + 1, 'model_state_dict': self.supernet.state_dict(), 'scheduler': self.scheduler,
                        'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(self.save_prefix, 'latest_model.pt'))

        total_sec = time.time() - train_start
        print('\n=== SuperNet training complete ===')
        print('Total time: %.1f s  (%.4f GPU-days)' % (total_sec, total_sec / 86400.0))

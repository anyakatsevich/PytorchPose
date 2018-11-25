import logging
import argparse
import time
import os
import pickle
import numpy as np
from logger import Logger
import sys
from collections import OrderedDict
from tensorboardX import SummaryWriter      

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network.rtpose_vgg import get_model, use_vgg

#####################################
#               DATASET             #
#####################################
class Pose_Dataset(Dataset):
    def __init__(self, data_dir, num_im):
        super(Pose_Dataset, self).__init__()
        data_files = os.listdir(data_dir)
        self.data_paths = [os.path.join(data_dir, d) for d in data_files]
        self.data_paths = self.data_paths[0:num_im]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        with open(self.data_paths[index], 'rb') as f:
            sample = pickle.load(f)
        return sample

logger = Logger('./logs')
#sys.argv=['']; 
#del sys

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch rtpose Training')

s = '/scratch/aek495/pyenv2/pytorch_Pose/'
parser.add_argument('--data_dir', default=s+'training/dataset/COCO/images/', type=str, metavar='DIR',
                    help='path to where coco images stored') 
parser.add_argument('--preproc_dir', default=s+'training/dataset/COCO/preprocess', type=str, metavar='DIR',
                    help='path to where coco images preprocessed') 
parser.add_argument('--valid_dir', default=s+'training/dataset/COCO/preprocess/valid', type=str, metavar='DIR',
                    help='path to preprocessed valid images') 
parser.add_argument('--train_dir', default=s+'training/dataset/COCO/preprocess/train', type=str, metavar='DIR',
                    help='path to preprocessed train images') 
parser.add_argument('--mask_dir', default=s+'training/dataset/COCO/mask/', type=str, metavar='DIR',
                    help='path to where coco images stored')    
parser.add_argument('--logdir', default=s+'logs/', type=str, metavar='DIR',
                    help='path to where tensorboard log restore')                                       
parser.add_argument('--json_path', default=s+'training/dataset/COCO/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')                                      
parser.add_argument('--model_path', default=s+'network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')                     
parser.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
 
parser.add_argument('--num_train', default=50, type=int, metavar='NT',
                    help='number of training images')

parser.add_argument('--num_valid', default=10, type=int, metavar='NV',
                    help='number of validation images')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
                    
parser.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  
parser.add_argument('--nesterov', dest='nesterov', action='store_true')     
                                                   
parser.add_argument('-o', '--optim', default='sgd', type=str)
#Device options
parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+",
                    default=[0,1,2,3], type=int)
                    
parser.add_argument('--batch_size', default=80, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')
args = parser.parse_args()  
               
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)

params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.5
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 40

# ===
params_transform['center_perterb_max'] = 40

# === aug_flip ===
params_transform['flip_prob'] = 0.5

params_transform['np'] = 56
params_transform['sigma'] = 7.0
params_transform['limb_width'] = 1.

def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, heat_weight,
               vec_temp, vec_weight):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    #criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=args.gpu_ids)
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j] * vec_weight
        """
        print("pred1 sizes")
        print(saved_for_loss[2*j].data.size())
        print(vec_weight.data.size())
        print(vec_temp.data.size())
        """
        gt1 = vec_temp * vec_weight

        pred2 = saved_for_loss[2 * j + 1] * heat_weight
        gt2 = heat_weight * heat_temp
        """
        print("pred2 sizes")
        print(saved_for_loss[2*j+1].data.size())
        print(heat_weight.data.size())
        print(heat_temp.data.size())
        """

        # Compute losses
        loss1 = criterion(pred1, gt1) * 0
       # if j == 0:
        loss2 = criterion(pred2, gt2) 
        #else:
         #   loss2 = criterion(pred2, gt2) * 0

        total_loss += loss1
        total_loss += loss2
        # print(total_loss)

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log
         

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    
    # switch to train mode
    model.train()

    end = time.time()
    print("training")
    nb_found = 0
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
        #print(img.size())
        if type(img[0]) == 'str':	
        #    print('in here')
            continue
        else:
            nb_found += 1
         #   print(nb_found, i)
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #  writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        img = img.squeeze(1).cuda()
        heatmap_target = heatmap_target.squeeze(1).cuda()
        heat_mask = heat_mask.squeeze(1).cuda()
        paf_target = paf_target.squeeze(1).cuda()
        paf_mask = paf_mask.squeeze(1).cuda()
        
        #img = img.squeeze(1)
        #heatmap_target = heatmap_target.squeeze(1)
        #heat_mask = heat_mask.squeeze(1)
        #paf_target = paf_target.squeeze(1)
        #paf_mask = paf_mask.squeeze(1)
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
               paf_target, paf_mask)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #print('epoch= '+str(epoch))
        #print('dataset length = ' + str(len(train_dataset))+ ' batch size = '+str(args.batch_size))
        step = epoch*len(train_dataset)/args.batch_size + i
        #print(step)
        logger.scalar_summary('train_loss',losses.avg,step)

        #if i % args.print_freq == 0:
         #   logging.info('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(train_loader)))
          #  logging.info('Data time {data_time.val:.3f} ({data_time.avg:.3f})'.format( data_time=data_time))
           # logging.info('Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

            #print_string = ''
            #for name, value in meter_dict.items():
             #   print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            #print(print_string)
    return losses.avg  
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    nb_found = 0
    end = time.time()
    print('validating')
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(val_loader):
        #print(img.size())
        if type(img[0]) == 'str':
            continue
        else:
            nb_found += 1
          #  print(nb_found, i)
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.squeeze(1).cuda()
        heatmap_target = heatmap_target.squeeze(1).cuda()
        heat_mask = heat_mask.squeeze(1).cuda()
        paf_target = paf_target.squeeze(1).cuda()
        paf_mask = paf_mask.squeeze(1).cuda()
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
               paf_target, paf_mask)
               
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
            
        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        #if i % args.print_freq == 0:
         #   logging.info('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))
          #  logging.info('Data time {data_time.val:.3f} ({data_time.avg:.3f})'.format(data_time=data_time))
           # logging.info('Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

            #print_string = ''
            #for name, value in meter_dict.items():
             #   print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            #print(print_string)
                
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


print("Loading dataset...")
# load data
# TODO update data loaders
valid_dataset = Pose_Dataset(args.valid_dir, args.num_valid)
train_dataset = Pose_Dataset(args.train_dir, args.num_train)
valid_loader = DataLoader(valid_dataset,
                          batch_size=args.batch_size,
                          num_workers=0)
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0)
print("{:6d} train samples".format(len(train_dataset)))
print("{:6d} valid samples".format(len(valid_dataset)))

# model
model = get_model(trunk='vgg19')
#model = encoding.nn.DataParallelModel(model, device_ids=args.gpu_ids)
#model = torch.nn.DataParallel(model)
model = torch.nn.DataParallel(model).cuda()
# load pretrained
use_vgg(model, args.model_path, 'vgg19')


# Fix the VGG weights first, and then the weights will be released
for i in range(20):
    for param in model.module.model0[i].parameters():
        param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)
 
print("Beginning training.")
print("WARNING: only non-zero loss is stage 1, branch 2")
start_time = time.time()
for epoch in range(5):
    logging.info("Epoch {}".format(epoch))
    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)
   
    # evaluate on validation set
    val_loss = validate(valid_loader, model, epoch)  
    logger.scalar_summary('Val_loss',val_loss,epoch) 
    t = (time.time()-start_time)/60.0
    print('Epoch '+str(epoch)+' took '+str(t)+ ' minutes ---')
    logging.info("Train loss: {:.6f}".format(train_loss))
    logging.info("Valid loss: {:.6f}".format(val_loss))
                                 
# Release all weights                                   
for param in model.module.parameters():
    param.requires_grad = True

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)          
                                                    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf


model_save_filename = './network/weight/best_pose_test.pth'
for epoch in range(5, args.epochs):

    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(valid_loader, model, epoch)   
    t = (time.time()-start_time)/60.0
    print('Epoch '+str(epoch)+' took '+str(t)+ ' minutes ---')

    logger.scalar_summary('Val_loss',val_loss,epoch) 
    logging.info("Train loss: {:.6f}".format(train_loss))
    logging.info("Valid loss: {:.6f}".format(val_loss))

    lr_scheduler.step(val_loss)                        
    
    is_best = val_loss<best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)      
        

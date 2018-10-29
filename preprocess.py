import argparse
import time
import os
import pickle


from training.datasets.coco import get_loader

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch rtpose Training')
s = '/mnt/c/Users/Anya/Documents/NYU/VisionML/Skeleton_Nick/'
parser.add_argument('--data_dir', default=s+'training/dataset/COCO/images/', type=str, metavar='DIR',
                    help='path to where coco images stored') 
parser.add_argument('--preproc_dir', default=s+'training/dataset/COCO/preprocess', type=str, metavar='DIR',
                    help='path to where coco images preprocessed') 
parser.add_argument('--mask_dir', default=s+'training/dataset/COCO/mask2014/', type=str, metavar='DIR',
                    help='path to where coco images stored')    
parser.add_argument('--json_path', default=s+'training/dataset/COCO/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')    
                                    

args = parser.parse_args()  
               

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

def preprocess_dataset(data_loader, savedir, nb_aug=1):
    for i in range(nb_aug):
        for j, sample in enumerate(data_loader):
            out_file = os.path.join(savedir,"event_{:02d}_{:06d}.pickle".format(i,j))
            with open(out_file, 'wb') as f:
                pickle.dump(sample, f)
            if (j % 500) == 0:
                print("{:2d}, {:6d}".format(i,j))
        

print("Loading dataset...")
# validation data
valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, 368,
                            8, preprocess='vgg', training=False,
                            batch_size=1, params_transform = params_transform, shuffle=False, num_workers=1)
print('val dataset len: {}'.format(len(valid_data.dataset)))
valid_savedir = os.path.join(args.preproc_dir, "valid")
os.makedirs(valid_savedir, exist_ok=True)
preprocess_dataset(valid_data, valid_savedir)
# load data
train_data = get_loader(args.json_path, args.data_dir,
                        args.mask_dir, 368, 8,
                        'vgg', 1, params_transform = params_transform, 
                        shuffle=True, training=True, num_workers=0)
print('train dataset len: {}'.format(len(train_data.dataset)))
train_savedir = os.path.join(args.preproc_dir, "train")
os.makedirs(train_savedir, exist_ok=True)
preprocess_dataset(train_data, train_savedir, nb_aug=4)

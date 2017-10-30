# use to train a DL model that learns where the action on a per-frame basis 
# or to generate (-gtc) videos/images of leveled and centered crops for test videos (loading a saved model)

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from collections import OrderedDict
import itertools

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import MeanAverageError
from torchsample import TensorDataset

import os
from torchvision import datasets, transforms, models

from fishdataset import BoatDataset, Resize, rotate, remove_orientation, SubsetRandomBalancedBatchSampler
from sklearn.model_selection import train_test_split
import math
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-gtc', '--generate-test-crops', action='store_true', help='Generate test crops loading model (videos and images)')
parser.add_argument('-gio', '--generate-images-only', action='store_true', help='When used with -gtc only generate images')
parser.add_argument('-lm', '--load-model', type=str, help='Load model from file')
parser.add_argument('-cs', '--crop-size', type=int, default = 384, help='Crop size')
parser.add_argument('-cm', '--crop-margin', type=int, default = 64, help='Crop margin (really half of this on each side)')

args = parser.parse_args()

CROP_SIZE   = args.crop_size
CROP_MARGIN = args.crop_margin  # it's really CROP_SIZE/2 on each side

args = parser.parse_args()

PATCH_SIZE = 224
FRAMES_PER_VIDEO = 2
TEST_FOLDER = 'test_videos'
TEST_CROPS_FOLDER = 'test_crops'
TEST_IMAGES_FOLDER = 'test_images'
BOAT_ANGLES_CSV = 'boat_angles.csv'

class BoatNet(nn.Module):
    def __init__(self):
        super(BoatNet, self).__init__()
        resnet_model = models.resnet152(pretrained=True)
        self.num_ftrs  = resnet_model.fc.in_features * 49
        resnet_dict = OrderedDict(resnet_model.named_children()) 
        resnet_dict.popitem(last=True) # remove 'fc' layer
        resnet_dict.popitem(last=True) # remove 'fc' layer
        self.features = nn.Sequential(
            resnet_dict
        )
        self.class_fc1     = nn.Linear(self.num_ftrs * FRAMES_PER_VIDEO, 512)
        self.class_fc2     = nn.Linear(512, 128)
        self.class_fc3     = nn.Linear(128, 64)
        self.class_xy      = nn.Linear(64, 2)
        #self.class_angle   = nn.Linear(64, 1)
        self.class_boat_id = nn.Linear(64, 5)

    def forward(self, x1, x2):
        if FRAMES_PER_VIDEO != 1:
            x = [x1, x2]
            x_c = None
            for x_it in x:
                x_it = self.features(x_it)
                x_it = x_it.view(-1,self.num_ftrs )
                if x_c is None:
                    x_c = x_it
                else:
                    x_c = th.cat([x_c, x_it], dim=1)
        else:
            x_c = self.features(x1)
            x_c = x_c.view(-1,self.num_ftrs )

        x_c = nn.Dropout(p=0.05)(x_c)
        x_c = F.relu(self.class_fc1(x_c))
        x_c = nn.Dropout(p=0.05)(x_c)
        x_c = F.relu(self.class_fc2(x_c))
        x_c = nn.Dropout(p=0.05)(x_c)
        x_c = F.relu(self.class_fc3(x_c))

        xy      =  self.class_xy(x_c)
        boat_id =  nn.LogSoftmax()(self.class_boat_id(x_c))
        return  xy, boat_id

model = BoatNet()
if th.cuda.is_available():
    model    = model.cuda()

print(model)

initial_epoch = 0

optimizer = optim.Adam([ 
    { 'params': model.class_fc1.parameters() }, 
    { 'params': model.class_fc2.parameters() }, 
    { 'params': model.class_fc3.parameters() }, 
    { 'params': model.class_xy.parameters() },
    { 'params': model.class_boat_id.parameters()},
    { 'params': model.features.layer4.parameters() },
    { 'params': model.features.layer3.parameters(), 'lr' : 1e-6 },
    { 'params': model.features.layer2.parameters(), 'lr' : 1e-6 },
    ], lr= 1e-5)

if args.load_model:
    print("Loading checkpoint: " + args.load_model)
    checkpoint = th.load(args.load_model, map_location=lambda storage, loc: storage)
    #print(checkpoint.keys())
    initial_epoch = checkpoint['epoch']
    #print(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if th.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

if args.generate_test_crops:
    test_dataset = BoatDataset(
        xy_angle_csv_file=None, 
        fish_csv_file = None,
        videos_dir=TEST_FOLDER,
        boat_frames_dir = None,
        frames_per_video = FRAMES_PER_VIDEO,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            Resize((PATCH_SIZE, PATCH_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.40056697,  0.39674244,  0.42981134],
                std=[0.27938687,  0.28158916,  0.29005027]
                #mean=[0.485, 0.456, 0.406],
                #std=[0.229, 0.224, 0.225]
                ),
            ])
        )

    test_loader = DataLoader(test_dataset,
                      batch_size=2,
                      num_workers=1,
                      pin_memory=True if th.cuda.is_available() else False)

    test_trainer = ModuleTrainer(model)

    predictions = test_trainer.predict_loader(
        test_loader,
        verbose=1, 
        cuda_device=0 if th.cuda.is_available() else -1)

    boat_angles_frame = pd.read_csv(BOAT_ANGLES_CSV).set_index('boat_id')  

    for video_id, xy, boat_id in tqdm(zip(test_dataset.test_video_ids, predictions[0].cpu().data.numpy().squeeze(), predictions[1].cpu().data.numpy())):

        boat_id = np.argmax(boat_id)
        angle = boat_angles_frame.get_value(boat_id, 'angle')[0]
        cx =    boat_angles_frame.get_value(boat_id, 'cx')[0]
        cy =    boat_angles_frame.get_value(boat_id, 'cy')[0]

        xy += np.float32((cx,cy))

        p0 = xy
        p1 = p0 + rotate(np.array([(CROP_SIZE-CROP_MARGIN)/2, 0.  ]), -angle) 
        p2 = p1 + rotate(np.array([0. ,(CROP_SIZE-CROP_MARGIN)/2]), -angle)

        dp = np.float32(([CROP_SIZE/2,CROP_SIZE/2], [CROP_SIZE - CROP_MARGIN/2,CROP_SIZE/2], [CROP_SIZE - CROP_MARGIN/2,CROP_SIZE - CROP_MARGIN/2])) 

        M = cv2.getAffineTransform(np.float32((p0,p1,p2)), dp)

        video_in  = cv2.VideoCapture(os.path.join(TEST_FOLDER, video_id) + '.mp4')
        if not args.generate_images_only:
            fourcc    = cv2.VideoWriter_fourcc(*'MJPG')
            video_out = cv2.VideoWriter(os.path.join(TEST_CROPS_FOLDER, video_id)  + '_' +str(int(angle)) + '.avi',fourcc, 15., (CROP_SIZE,CROP_SIZE))

        rot_video = np.empty([CROP_SIZE,CROP_SIZE,3], dtype=np.float32)

        it = 0
        while(video_in.isOpened()):
            ret, frame = video_in.read()
            if ret==True:
                rot_video = cv2.warpAffine(frame,M,(CROP_SIZE,CROP_SIZE))
                if not args.generate_images_only:
                    video_out.write(rot_video)
                cv2.imwrite(os.path.join(TEST_IMAGES_FOLDER, video_id) + '_' + str(it) + '.png', rot_video)
                it += 1
            else:
                break
        video_in.release()
        if not args.generate_images_only:
            video_out.release()
else:

    dataset = BoatDataset(
        xy_angle_csv_file='training_transform.csv', 
        fish_csv_file='training.csv', 
        boat_frames_dir='train_boats',
        videos_dir = None,
        frames_per_video = FRAMES_PER_VIDEO,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            Resize((PATCH_SIZE, PATCH_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.40056697,  0.39674244,  0.42981134],
                std=[0.27938687,  0.28158916,  0.29005027]
                #mean=[0.485, 0.456, 0.406],
                #std=[0.229, 0.224, 0.225]
                ),
            ])
        )

    idx_train, idx_valid = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_sampler = SubsetRandomBalancedBatchSampler(idx_train[:], dataset.boat_ids, batch_size=16) #SubsetRandomSampler(idx_train[:])
    valid_sampler = SubsetRandomSampler(idx_valid[:])

    train_loader = DataLoader(dataset,
                          #batch_size=8,
                          batch_sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)

    valid_loader = DataLoader(dataset,
                          batch_size=8,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in itertools.chain(model.features.layer4.parameters(), model.features.layer3.parameters(), model.features.layer2.parameters()):
        param.requires_grad = True

    trainer = ModuleTrainer(model)

    callbacks = [ 
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(
            directory='checkpoints',
            filename='boatnet_{epoch}_{loss}.pth',
            save_best_only=True,
            verbose=1) 
        ]

    initializers = [XavierUniform(bias=False, module_filter='class*')]
    metrics = [[MeanAverageError(), MeanAverageError()]]

    def angle_loss(input, target):
        # input is [-1,1] target is [-0.5, 0.5]

        input  *= math.pi
        target *= math.pi
        input  = input.squeeze()
        target = target.squeeze()
        _target = target + math.pi
        #print(input * 180/math.pi, target * 180/math.pi)
        
        input_unit_angle   = th.stack([  input.cos(),  input.sin()], dim=1)
        target_unit_angle  = th.stack([ target.cos(), target.sin()], dim=1)
        _target_unit_angle = th.stack([_target.cos(),_target.sin()], dim=1)

        dist_A = nn.PairwiseDistance()(input_unit_angle,  target_unit_angle)
        dist_B = nn.PairwiseDistance()(input_unit_angle, _target_unit_angle)

        dist = th.min(dist_A, dist_B)

        return 1. * dist.mean() 

    def null_loss(input, target):
        return 0.

    def div_mse_loss(input, target):
        return 1e-2 * nn.MSELoss()(input, target)

    def mul_nll_loss(input, target):
        return 1e2  *  F.nll_loss(input, target)

    trainer.compile(loss=[div_mse_loss, mul_nll_loss],
                    optimizer=optimizer,
                    #regularizers=regularizers,
                    #constraints=constraints,
                    #metrics=metrics,
                    initializers=initializers,
                    callbacks=callbacks)

    trainer.fit_loader(train_loader, valid_loader, initial_epoch=initial_epoch, num_epoch=200, verbose=1, cuda_device=0 if th.cuda.is_available() else -1)



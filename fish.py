
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
from torchsample.metrics import CategoricalAccuracy
from torchsample import TensorDataset
from torchsample.transforms.affine_transforms import RandomRotate
from torchsample.transforms.tensor_transforms import RandomCrop, RandomFlip, SpecialCrop

import os
from torchvision import datasets, transforms, models

from fishdataset import FishDataset, SubsetRandomBalancedBatchSampler, SubsetSampler
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test', action='store_true', help='Test model')
parser.add_argument('-lm', '--load-model', type=str, help='Load model from file')

args = parser.parse_args()

PATCH_SIZE = 224

class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        resnet_model = models.resnet152(pretrained=True)
        self.num_ftrs  = resnet_model.fc.in_features
        resnet_dict = OrderedDict(resnet_model.named_children()) 
        resnet_dict.popitem(last=True) # remove 'fc' layer
        self.features = nn.Sequential(
            resnet_dict
        )
        self.class_fc1 = nn.Linear(self.num_ftrs, 512)
        self.class_fc2 = nn.Linear(512, 9)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,self.num_ftrs )
        species_length = F.relu(self.class_fc1(x))
        species_length = nn.Dropout(p=0.05)(species_length)
        species_length = self.class_fc2(species_length)
        _species = species_length.clone()[:,:8,...]
        _length  = species_length.clone()[:, 8,...]
        species_length[:,:8,...] = nn.LogSoftmax()(_species)
        species_length[:, 8,...] = F.relu(_length)
        return species_length, th.autograd.Variable(th.FloatTensor([0.]))

model = FishNet()
#print(model)

initial_epoch = 0

for param in model.features.parameters():
    param.requires_grad = False

for param in itertools.chain(model.features.layer3.parameters(), model.features.layer4.parameters()):
    param.requires_grad = True

optimizer = optim.SGD([ 
    { 'params': model.class_fc1.parameters() }, 
    { 'params': model.class_fc2.parameters() }, 
    { 'params': model.features.layer4.parameters() },
    { 'params': model.features.layer3.parameters() }, 
    ], lr= 5e-4, momentum=0.9)

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

if args.test:

    test_dataset = FishDataset(
        csv_file= None,
        images_dir= None,
        videos_dir= 'train_crops',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(PATCH_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.55803171,  0.54565148,  0.57604222],
                std=[0.30346842,  0.3020077 ,  0.30457914]
                ),
            ])
        )

    test_loader = DataLoader(
        test_dataset,
        sampler = SubsetSampler(20),
        batch_size=64,
        num_workers=1,
        pin_memory=True if th.cuda.is_available() else False)

    test_trainer = ModuleTrainer(model)

    predictions = test_trainer.predict_loader(
        test_loader,
        verbose=1, 
        cuda_device=0 if th.cuda.is_available() else -1)

    import pandas as pd
    print(predictions[0])
    fish_frame = pd.DataFrame(predictions, columns = [
        'row_id',
        'video_id',
        'frame',
        'fish_number',
        'length',
        'species_fourspot',
        'species_grey sole',
        'species_other',
        'species_plaice',
        'species_summer',
        'species_windowpane',
        'species_winter'])


else:

    dataset = FishDataset(
        csv_file='training.csv', 
        images_dir='train_images', 
        transform=transforms.Compose([
            transforms.RandomCrop(PATCH_SIZE),
            transforms.ToTensor(),
            RandomFlip(True,True),
            RandomRotate(2.),
            transforms.Normalize(
                mean=[0.55803171,  0.54565148,  0.57604222],
                std=[0.30346842,  0.3020077 ,  0.30457914]
                ),
            ])
        )

    idx_train, idx_valid = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
    train_sampler = SubsetRandomBalancedBatchSampler(idx_train[:], dataset.classes, batch_size=24)
    valid_sampler = SubsetRandomSampler(idx_valid[:])

    train_loader = DataLoader(dataset,
                          #batch_size=24,
                          batch_sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)

    valid_loader = DataLoader(dataset,
                          batch_size=32,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)


    trainer = ModuleTrainer(model)

    callbacks = [ 
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(
            directory='checkpoints',
            filename='fishnet_{epoch}_{loss}.pth',
            save_best_only=True,
            verbose=1) 
        ]

    initializers = [XavierUniform(bias=False, module_filter='class*')]
    metrics = [CategoricalAccuracy()]

    def species_length_loss(input, target):

        input_species = input[:,:8]
        input_length  = input[:, 8]

        target_species = target[:,0].long()
        target_length  = target[:,1]

        input_length = input_length * (target_species != 7).float()

        loss = nn.MSELoss()(input_length, target_length) * 1e-5 + F.nll_loss(input_species, target_species) 

        return loss

    def null_loss(input, target):
        return th.autograd.Variable(th.FloatTensor([0.])).cuda()

    trainer.compile(loss=[species_length_loss, null_loss],
                    optimizer=optimizer,
                    #regularizers=regularizers,
                    #constraints=constraints,
                    #metrics=metrics,
                    initializers=initializers,
                    callbacks=callbacks,
                    #transforms=transforms.Compose([RandomRotate(5.)]),
    )

    trainer.fit_loader(
        train_loader, 
        valid_loader, 
        initial_epoch=initial_epoch, 
        num_epoch=200, 
        verbose=1, 
        cuda_device=0 if th.cuda.is_available() else -1)
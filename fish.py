
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
import pandas as pd

from se_resnet import se_resnet152

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test-video-directory', type=str, help='Test model on directory e.g. -t train_crops')
parser.add_argument('-b', '--use-boat-ids', action='store_true', help='Use boat_angles.csv for stratified (boat/species) class balancing')
parser.add_argument('-lm', '--load-model', type=str, help='Load model from file')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('-s', '--suffix',  type=str, default=None, help='Suffix to store checkpoints')
parser.add_argument('-se', '--squeeze-excitation',  action='store_true', help='Use SE resnet')
parser.add_argument('-bs', '--batch-size', type=int, default=24, help='Batch size')

args = parser.parse_args()

PATCH_SIZE = 224

class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        if args.squeeze_excitation:
            resnet_model = se_resnet152(10)
        else:
            resnet_model = models.resnet152(pretrained=True)
        #pool_factor    = 7
        self.num_ftrs  = resnet_model.fc.in_features# * 49 / (pool_factor ** 2)
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
print(model)

initial_epoch = 0

for param in model.features.parameters():
    param.requires_grad = False

#for param in itertools.chain(model.features.layer3.parameters(), model.features.layer4.parameters()):
for param in itertools.chain(model.features.parameters()):
    param.requires_grad = True

optimizer = optim.SGD([ 
    { 'params': model.parameters() }, 
#    { 'params': model.class_fc1.parameters() }, 
#    { 'params': model.class_fc2.parameters() }, 
    # { 'params': model.class_fc3.parameters() }, 
#    { 'params': model.features.layer4.parameters() },
#    { 'params': model.features.layer3.parameters(), 'lr' : 5e-4 }, 
    #{ 'params': model.features.layer2.parameters(), 'lr' : 1e-4 }, 
    ], lr= args.learning_rate, momentum=0.9)

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

if args.test_video_directory:

    test_dataset = FishDataset(
        csv_file= None,
        images_dir= None,
        videos_dir= args.test_video_directory,
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
        #sampler = SubsetSampler(100),
        batch_size=64,
        num_workers=1,
        pin_memory=True if th.cuda.is_available() else False)

    test_trainer = ModuleTrainer(model)

    predictions = test_trainer.predict_loader(
        test_loader,
        verbose=1, 
        cuda_device=0 if th.cuda.is_available() else -1)

    species = predictions[0].data[:,:8].topk(8)

    species_softmax = predictions[0].data[:,:8].exp()
    ll = len(predictions[0])
    print(ll)

    video_filename__base_frame = test_dataset.video_index_frame.ix[range(ll), 
        [ test_dataset.video_index_frame.columns.get_loc('video_id'), test_dataset.video_index_frame.columns.get_loc('base_frame') ] ].values

    length = predictions[0].data[:,8].cpu().numpy()
    species_fourspot__species_no_fish = species_softmax.cpu().numpy()

    print(video_filename__base_frame)

    fish_frame = pd.DataFrame(columns = [
        'video_id',
        'frame',
        'length',
        'species_fourspot',
        'species_grey sole',
        'species_other',
        'species_plaice',
        'species_summer',
        'species_windowpane',
        'species_winter',
        'species_no_fish'], index=range(ll))

    fish_frame.index.name = 'row_id'

    fish_frame.loc[range(ll),'video_id':'frame'] = video_filename__base_frame
    fish_frame.loc[range(ll),'length'] = length
    fish_frame.loc[range(ll),'species_fourspot':'species_no_fish'] = species_fourspot__species_no_fish

    fish_frame.to_csv(os.path.basename(args.test_video_directory) + '.csv')

# ['bc6hkwua3iLReunk', '8jkQWJWPCtIvcnmH', 'tJinkrdMMZ477RGi', 'P3QkoeOjxoM6pDKb', 'pGd0FSJQcDH5DI8x', 'Sw0AgnH8BY1BDGHu', 'ZU6XtvFk0UMrHLEL', 'LU2DSX6VZcIsiyaW']
# and they are off by [1, 1, -3, 172, 1, 1, 1, 1] ('sfz' - 'actual frames').
# https://community.drivendata.org/t/frame-number-miss-match-in-test-set/1493

else:

    dataset = FishDataset(
        csv_file='training.csv', 
        images_dir='train_images',
        boat_ids_csv='boat_angles.csv' if args.use_boat_ids else None,
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
    train_sampler = SubsetRandomBalancedBatchSampler(idx_train[:], dataset.classes, batch_size=args.batch_size)
    valid_sampler = SubsetRandomSampler(idx_valid[:])

    train_loader = DataLoader(dataset,
                          batch_sampler=train_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)

    valid_loader = DataLoader(dataset,
                          batch_size=args.batch_size,
                          sampler=valid_sampler,
                          num_workers=4,
                          pin_memory=True if th.cuda.is_available() else False)


    trainer = ModuleTrainer(model)

    suffix = '-' + args.suffix if args.suffix else ''

    if args.squeeze_excitation:
        suffix = '-se' + suffix

    callbacks = [ 
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(
            directory='checkpoints',
            filename='fishnet'+suffix+'_{epoch}_{loss}.pth',
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
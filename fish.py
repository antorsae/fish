
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim


from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy
from torchsample import TensorDataset

import os
from torchvision import datasets, transforms, models

from fishdataset import FishDataset
from sklearn.model_selection import train_test_split

PATCH_SIZE = 224

dataset = FishDataset(
    csv_file='training.csv', 
    images_dir='train_images', 
    transform=transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
        ])
    )
idx_train, idx_valid = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
train_sampler = SubsetRandomSampler(idx_train)
valid_sampler = SubsetRandomSampler(idx_valid)

train_loader = DataLoader(dataset,
                      batch_size=32,
                      sampler=train_sampler,
                      num_workers=4,
                      pin_memory=True)

valid_loader = DataLoader(dataset,
                      batch_size=32,
                      sampler=valid_sampler,
                      num_workers=1,
                      pin_memory=True)

class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        self.resnet_model = models.resnet152(pretrained=True)
        self.num_ftrs  = self.resnet_model.fc.in_features
        self.features = nn.Sequential(
            *list(self.resnet_model.children())[:-1]
        )
        self.class_fc1 = nn.Linear(self.num_ftrs, 512)
        self.class_fc2 = nn.Linear(512, 9)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,self.num_ftrs )
        species_length = F.relu(self.class_fc1(x))
        species_length = self.class_fc2(species_length)
        _species = species_length.clone()[:,:8,...]
        _length  = species_length.clone()[:, 8,...]
        species_length[:,:8,...] = nn.LogSoftmax()(_species)
        species_length[:, 8,...] = F.relu(_length)
        return species_length

model = FishNet()
print(model)

#model    = models.resnet152(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

#for param in model.layer4.parameters():
#    param.requires_grad = True
#for param in model.layer3.parameters():
#    param.requires_grad = True

#num_ftrs  = model.fc.in_features

#class_head = list(model.fc.children())
#class_head.append(nn.Linear(num_ftrs, 512))
#class_head.append(nn.ReLU())
#class_head.append(nn.Linear(512, 8))
#class_head.append(nn.LogSoftmax())
#model.fc = nn.Sequential(*class_head)

model    = model.cuda()

trainer = ModuleTrainer(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                L2Regularizer(scale=1e-5, module_filter='fc*')]
constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*')]
initializers = [XavierUniform(bias=False, module_filter='fc*')]
metrics = [CategoricalAccuracy()]

optimizer = optim.Adam([ 
    { 'params': model.class_fc1.parameters() }, 
    { 'params': model.class_fc2.parameters() },  ], lr= 1e-2)
#    { 'params': model.layer4.parameters(), 'lr': 1e-3 },
#    { 'params': model.layer3.parameters(), 'lr': 1e-4 }, ], lr= 1e-2)

def species_length_loss(input, target):

    input_species = input[:,:8]
    input_length  = input[:, 8]

    target_species = target[:,0].long()
    target_length  = target[:,1]

    input_length = input_length * (target_species != 7).float()

    loss = nn.MSELoss()(input_length, target_length) * 1e-5 + F.nll_loss(input_species, target_species) 

    return loss

trainer.compile(loss=species_length_loss,
                optimizer=optimizer,
                #regularizers=regularizers,
                #constraints=constraints,
                #metrics=metrics,
                initializers=initializers,
                callbacks=callbacks)

trainer.fit_loader(train_loader, valid_loader, num_epoch=200, verbose=1, cuda_device=0)




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


# Define your model EXACTLY as if you were using nn.Module
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*(((((PATCH_SIZE-2)//2) - 2)//2)**2), 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64*(((((PATCH_SIZE-2)//2) - 2)//2)**2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.LogSoftmax()(x)




class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        self.resnet_model = models.resnet152(pretrained=True)
        self.num_ftrs  = self.resnet_model.fc.in_features
        self.features = nn.Sequential(
            *list(self.resnet_model.children())[:-1]
        )
        self.class_fc1 = nn.Linear(self.num_ftrs, 512)
        self.class_fc2 = nn.Linear(512, 8)
        self.reg_fc1 = nn.Linear(self.num_ftrs, 512)
        self.reg_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,self.num_ftrs )
        species = F.relu(self.class_fc1(x))
        species = nn.LogSoftmax()(self.class_fc2(species))
        length  = F.relu(self.reg_fc1(x))
        length  = F.relu(self.reg_fc2(length))
        return species, length

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
#print(model)

#model = Network().cuda()
#print(model)

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
    { 'params': model.class_fc2.parameters() }, 
    { 'params': model.reg_fc1.parameters() }, 
    { 'params': model.reg_fc2.parameters() }, ], lr= 1e-2)
#    { 'params': model.layer4.parameters(), 'lr': 1e-3 },
#    { 'params': model.layer3.parameters(), 'lr': 1e-4 }, ], lr= 1e-2)

trainer.compile(loss=['nll_loss', 'mse_loss'],
                optimizer=optimizer,
                #regularizers=regularizers,
                #constraints=constraints,
                #metrics=metrics,
                initializers=initializers,
                callbacks=callbacks)

trainer.fit_loader(train_loader, valid_loader, num_epoch=200, verbose=1, cuda_device=0)



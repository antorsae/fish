from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our face landmarks dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#
# Sample of our dataset will be a dict
# ``{'image': image, 'landmarks': landmarks}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
#

class FishDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fish_frame = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform  = transform
        self.num_inputs  = 1
        self.num_targets = 1

    def __len__(self):
        return len(self.fish_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.fish_frame.ix[idx, 'video_id']) + "_" + str(self.fish_frame.ix[idx, 'frame']) + ".png"
        image = Image.open(img_name)

        length = np.nan_to_num(self.fish_frame.ix[idx, 'length'].astype(np.float32).reshape(1,-1))
        
        species = np.empty((8,), dtype='int')
        species[:7] = self.fish_frame.ix[idx, 'species_fourspot':].as_matrix().astype('int')
        species[7]  = 1 if np.all(species[:7] == 0) else 0
        species = np.argmax(species)

        species_length = np.float32([species, length])

        if self.transform:
            image = self.transform(image)


        #species_length = (species, length)
        return image, species_length

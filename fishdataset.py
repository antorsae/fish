from __future__ import print_function, division
import os
import torch as th
import pandas as pd
#from skimage import io, transform
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
import glob
import iterm
import random
import copy

import cv2

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

SX = 1280
SY =  720

X_MARGIN = 55
Y_MARGIN = 20

def rotate(a, angle):
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta),  np.cos(theta)]])
    return np.dot(a, rotMatrix)

def remove_orientation(yaw):
    yaw = np.fmod(yaw, 180.)
    if yaw >= 90.:
        yaw -= 180.
    elif yaw <= -90.:
        yaw += 180.
    assert (yaw <= (90.)) and (yaw >= (-90.))
    return yaw

######################################################################

class SubsetRandomBalancedBatchSampler(Sampler):
    """Samples elements randomly from a given list of indices yielding
    batches containing samples of the same class, eventually repeating
    elements if needed so that total sampled classes are balanced as
    described in Class-aware sampling here:

    Relay Backpropagation for Effective Learning of Deep Convolutional 
    Neural Networks - https://arxiv.org/abs/1512.05830

    Arguments:
        indices (list): a list of indices to draw from (subset of dataset)
        class_vector (array-like): target classes for each index of dataset
        batch_size (integer)
    """

    def __init__(self, indices, class_vector, batch_size):
        self.indices = indices
        self.class_vector = class_vector
        self.batch_size = batch_size

        self.classes = np.unique(self.class_vector)
        self.class_map = { }
        self.running_class_map = { }
        for it in self.classes:
            self.class_map[it] = np.intersect1d(self.indices, np.where(self.class_vector == it)[0]).tolist()
            self.running_class_map[it] = copy.copy(self.class_map[it])
            random.shuffle(self.running_class_map[it])

    def __iter__(self): 
        batch = []
        for it in range(len(self.indices)):
            if True: #it % self.batch_size == 0:
                c = int(np.random.choice(self.classes, 1))
            l = self.running_class_map[c]
            if len(l) > 0:
                batch.append(l.pop())
            else:
                self.running_class_map[c] = copy.copy(self.class_map[c])
                random.shuffle(self.running_class_map[c])
                l = self.running_class_map[c]
                batch.append(l.pop())
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        #return len(self.indices)
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

######################################################################

class SubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, n_indices):
        self.n_indices = n_indices

    def __iter__(self):
        return iter(range(self.n_indices))

    def __len__(self):
        return self.n_indices

######################################################################

class FishDataset(Dataset):

    def __init__(self, csv_file, images_dir, videos_dir=None, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_inputs  = 1
        self.images_dir = images_dir
        self.fish_frame = None

        if csv_file:
            self.num_targets = 2
            self.fish_frame = pd.read_csv(csv_file)

            classes = []
            species = np.empty((8,), dtype='int')
            for idx in range(len(self.fish_frame)):
                species[:7] = self.fish_frame.ix[idx, 'species_fourspot':].as_matrix().astype('int')
                species[7]  = 1 if np.all(species[:7] == 0) else 0
                classes.append(np.argmax(species))
            self.classes = np.int32(classes)
        else:
            self.num_targets = 0
            if images_dir:
                s = lambda x : [x[0], int(x[1])]
                self.fish_frame = pd.DataFrame(sorted(map(s, [os.path.basename(x)[:-4].split('_')[0:2] for x in glob.glob(os.path.join(self.images_dir, '*.png'))])), 
                    columns=['video_id', 'frame'])

        if self.fish_frame is not None:
            self.total_images = len(self.fish_frame)

        if videos_dir:
            self.videos_dir = videos_dir
            self.total_images = 0
            self.video_frames = [ ] 
            for video_filename in sorted(glob.glob(os.path.join(self.videos_dir, '*.avi'))):
                vin = cv2.VideoCapture(video_filename)
                frames = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
                vin.release()
                self.video_frames.append((self.total_images, video_filename))
                self.total_images += frames
            self.video_frames_frame = pd.DataFrame(self.video_frames, columns=['frames', 'video_filename']).set_index('frames')

            self.video_filename = None
            self.vin = None

        self.transform  = transform
    
    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        if self.fish_frame is not None:
            img_name = os.path.join(self.images_dir, self.fish_frame.ix[idx, 'video_id']) + "_" + str(self.fish_frame.ix[idx, 'frame']) + ".png"
            image = Image.open(img_name)
        else:
            assert self.video_frames_frame is not None
            index_into_frame = self.video_frames_frame.index.get_loc(idx, 'pad')
            base_frame = self.video_frames_frame.index.values[index_into_frame]
            video_filename = self.video_frames_frame.iloc[index_into_frame, 0] # tidy up with label column
            frame_number = idx - base_frame
            if (video_filename == self.video_filename) and self.vin is not None:
                pass # do nothing
            else:
                if self.vin is not None:
                    self.vin.release()
                self.vin = cv2.VideoCapture(video_filename)
                self.video_filename = video_filename

            current_frame = int(self.vin.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame != frame_number:
                self.vin.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                assert frame_number == int(self.vin.get(cv2.CAP_PROP_POS_FRAMES))

            assert self.vin.isOpened()

            ret, image = self.vin.read()
            assert ret==True
            image = image[...,::-1] 

        if self.transform:
            image = self.transform(image)

        if self.num_targets == 0:
            return image

        length = np.nan_to_num(self.fish_frame.ix[idx, 'length'].astype(np.float32).reshape(1,-1))
        
        species = np.empty((8,), dtype='int')
        species[:7] = self.fish_frame.ix[idx, 'species_fourspot':].as_matrix().astype('int')
        species[7]  = 1 if np.all(species[:7] == 0) else 0
        species = np.argmax(species)

        species_length = np.float32([species, length])

        return image, (species_length, 0.)

class BoatDataset(Dataset):

    def __init__(self, xy_angle_csv_file, fish_csv_file, boat_frames_dir, videos_dir, frames_per_video, transform=None, augment=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations or none for test.
            root_dir (string): Directory with all the images.
            frames_per_video(int): number of frames to render from each video (starting from 0)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_targets = 2 if xy_angle_csv_file else 0
        
        # contains angle and x y of average head position
        self.boat_frame = pd.read_csv(xy_angle_csv_file).set_index('video_id') if xy_angle_csv_file else None

        if fish_csv_file:
            self.fish_frame = pd.read_csv(fish_csv_file).dropna().reset_index(drop=True)
            self.fish_frame.set_index(['video_id','frame'],inplace=True)
        else:
            self.fish_frame = None

        self.boat_frames_dir = boat_frames_dir

        self.boat_frames = sorted([os.path.basename(x)[:-6].split('_')[0:2] for x in glob.glob(os.path.join(boat_frames_dir, '*_f.png'))]) if self.num_targets != 0 else None
        
        if self.num_targets == 2:
            self.boat_ids = []
            for (video_id, frame_id) in self.boat_frames:
                self.boat_ids.append(int(self.boat_frame.loc[video_id, 'boat_id']))

        self.videos_dir = videos_dir
        self.test_video_ids = [os.path.basename(x).split('.')[0] for x in glob.glob(os.path.join(videos_dir, '*.mp4'))] if self.num_targets == 0 else None
        self.transform  = transform

        self.num_inputs  = frames_per_video
        self.augment = augment

    def __len__(self):
        return len(self.test_video_ids) if self.num_targets == 0 else len(self.boat_frames)

    def __getitem__(self, idx): 

        if self.num_targets == 0:
            video_id =  self.test_video_ids[idx] 

            video_name = os.path.join(self.videos_dir, video_id) + ".mp4"

            video_in  = cv2.VideoCapture(video_name)
            it = 0
            frames = []
            x_offset = y_offset = 0
            while(video_in.isOpened() and it < self.num_inputs):
                ret, frame = video_in.read()
                if ret==True:
                    frames.append(frame[...,::-1])
                else:
                    break
                it += 1
            video_in.release()
        else:

            video_id, frame_id = self.boat_frames[idx]
            frames = []
            frame_id = int(frame_id)

            if True:
                # take the actual center of the fish for this FRAME
                #x1 = self.fish_frame.xs([video_id, frame_id])['x1'] 
                #x2 = self.fish_frame.xs([video_id, frame_id])['x2']
                #y1 = self.fish_frame.xs([video_id, frame_id])['y1']
                #y2 = self.fish_frame.xs([video_id, frame_id])['y2'] 
                #x = (x1 + x2) / 2.
                #y = (y1 + y2) / 2.
                x = self.boat_frame.loc[video_id, 'dx'].astype(np.float32).reshape(1,-1)
                y = self.boat_frame.loc[video_id, 'dy'].astype(np.float32).reshape(1,-1)
            else:
                # take the averaged center of the fish for this VIDEO
                x = self.boat_frame.loc[video_id, 'x'].astype(np.float32).reshape(1,-1)
                y = self.boat_frame.loc[video_id, 'y'].astype(np.float32).reshape(1,-1)

            xo,yo = x,y

            SCALE = 4
            if self.augment:

                random_angle = 0 #np.random.randint(180)

                while True:
                    x_offset = np.random.randint(SX)
                    xc = (x - x_offset) % SX
                    if (xc > X_MARGIN) and ((SX-xc) > X_MARGIN):
                        break
                while True:
                    y_offset = np.random.randint(SY)
                    yc = (y - y_offset) % SY
                    if (yc > Y_MARGIN) and ((SY-yc) > Y_MARGIN):
                        break
                x,y = xc, yc

                flipx = np.random.randint(2)
                flipy = np.random.randint(2)

                x_step = -1 if flipx else 1
                y_step = -1 if flipy else 1

                x = (x * x_step) % SX
                y = (y * y_step) % SY

            else:
                x_offset = y_offset = 0
                x_step = y_step = 1
                random_angle = 0

            for it in range(self.num_inputs):

                suffix = "_f" if it == 0 else ""
                filename = os.path.join(self.boat_frames_dir, video_id + "_" + str(frame_id + it) + suffix + ".png")
                frame = cv2.imread(filename, cv2.IMREAD_COLOR)[::SCALE,::SCALE,::-1] # GBR -> RGB

                def rotate_image(image, angle, image_center):
                  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
                  result = cv2.warpAffine(image, rot_mat, image.shape[:2][::-1],flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                  return result

                if random_angle != 0:
                    frame = rotate_image(frame, random_angle, (xo/SCALE,yo/SCALE))

                frame = np.pad(
                    frame, 
                    ((0,y_offset//SCALE),(0,x_offset//SCALE),(0,0)), 'wrap')[y_offset//SCALE : y_offset//SCALE + SY//SCALE, \
                                                                             x_offset//SCALE : x_offset//SCALE + SX//SCALE, \
                                                                             :][::y_step, ::x_step, :]
                frames.append(frame)

                # change to True to view inline with iterm
                if False and np.random.randint(50) == 0:
                    frame[:,int(x//SCALE)-2:int(x//SCALE)+2] = (0,255,0)
                    frame[int(y//SCALE)-2:int(y//SCALE)+2,:] = (0,255,0)
                    iterm.show_image(frame[:,:,:])

        if self.transform:
            for it, frame in enumerate(frames):
                frames[it] = self.transform(frame)

        if len(frames) == 1:
            frames = frames[0]

        if self.num_targets == 0:
            return frames
    
        angle = self.boat_frame.loc[video_id, 'angle'].astype(np.float32).reshape(1,-1) # -90, 90
        angle = remove_orientation(angle - random_angle)

        angle = x_step * y_step * angle

        xy = np.float32([x / 1., y / 1.])
        #xy = np.float32([x / SX - 0.5, y / SY - 0.5])

        return frames, ( xy, self.boat_ids[idx]) # -0.5, 0.5, -0.5, 0.5, [0..4]



# ------------------
from PIL import Image, ImageOps, ImageEnhance
import collections

class Resize(object):
    """Resize the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        return resize(img, self.size, self.interpolation)

def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL.Image to the given size.
    Args:
        img (PIL.Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL.Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)




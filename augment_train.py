import pandas as pd
from os.path import join
import numpy as np
from tqdm import tqdm

import cv2

TRAIN_FOLDER        = 'train_videos'

# mkdir -p first!
TRAIN_CROPS_FOLDER  = 'train_crops'
TRAIN_IMAGES_FOLDER = 'train_images'

PATCH_SIZE   = 384
PATCH_MARGIN = 64  # it's really PATCH_MARGIN/2 on each side

def angle_diff(x, y):
    return np.arctan2(np.sin(x-y), np.cos(x-y))

def rotate_image(image, angle, pivot):
    rot_mat = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1])


def rotate(a, angle):
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta),  np.cos(theta)]])
    return np.dot(a, rotMatrix)

frame = pd.read_csv('training.csv')
for video_id, video_frame in tqdm(frame.groupby(['video_id'])):
    # estimate mean of head and tail and termine based on variance which one is head (less variance)
    p1 = video_frame[['x1', 'y1']]
    p2 = video_frame[['x2', 'y2']]

    p1_mse = np.linalg.norm(p1 - p1.mean(), axis=1)
    p2_mse = np.linalg.norm(p2 - p2.mean(), axis=1)

    head_first = True
    if np.amax(p1_mse) > np.amax(p2_mse):
        head, tail = p2, p1
        head_first = False
    else:
        head, tail = p1, p2

    head.columns = tail.columns = ['x', 'y']

    vx,vy = (tail-head).mean()
    angle  = np.arctan2(vy,vx) * 180 / np.pi
    angles = np.arctan2((tail-head)['y'], (tail-head)['x']) * 180 / np.pi
    angle_error = np.absolute(np.amax(angle_diff(angles,angle)))

    hx,hy = head.mean()

    head_edge = np.array((hx,hy))

    p0 = head_edge
    p1 = p0 + rotate(np.array([PATCH_SIZE-PATCH_MARGIN, 0.  ]), -angle) 
    p2 = p1 + rotate(np.array([0. ,(PATCH_SIZE-PATCH_MARGIN)/2]), -angle)

    dp = np.float32(([PATCH_MARGIN/2,PATCH_SIZE/2], [PATCH_SIZE - PATCH_MARGIN/2,PATCH_SIZE/2], [PATCH_SIZE - PATCH_MARGIN/2,PATCH_SIZE - PATCH_MARGIN/2])) 

    M = cv2.getAffineTransform(np.float32((p0,p1,p2)), dp)

    video_in  = cv2.VideoCapture(join(TRAIN_FOLDER, video_id) + '.mp4')
    fourcc    = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = cv2.VideoWriter(join(TRAIN_CROPS_FOLDER, video_id)  + '-' + str(head_first) + '_' +str(int(angle)) + '.avi',fourcc, 15., (PATCH_SIZE,PATCH_SIZE))
    rot_video = np.empty([PATCH_SIZE,PATCH_SIZE,3], dtype=np.float32)
    it = 0
    while(video_in.isOpened()):
        ret, frame = video_in.read()
        if ret==True:
            rot_video = cv2.warpAffine(frame,M,(PATCH_SIZE,PATCH_SIZE))
            video_out.write(rot_video)
            if (video_frame['frame'] == it).any():
                cv2.imwrite(join(TRAIN_IMAGES_FOLDER, video_id)  + '_' + str(it) + '.png', rot_video)
        else:
            break
        it += 1
    video_in.release()
    video_out.release()


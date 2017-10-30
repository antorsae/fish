# use this to generate centered and leveled videos/imagesets and CSV with boat ids,xy centroids and rotation angles

import pandas as pd
from os.path import join
import numpy as np
from tqdm import tqdm
import argparse
import glob
import os

import cv2

TRAIN_FOLDER        = 'train_videos'

# mkdir -p first!
TRAIN_CROPS_FOLDER  = 'train_crops'
TRAIN_IMAGES_FOLDER = 'train_images'

TRAIN_BOATS_FOLDER  = 'train_boats'

BOAT_ANGLES_CSV = 'boat_angles.csv'

parser = argparse.ArgumentParser()

parser.add_argument('-gv', '--generate-video', action='store_true', help='Generate framed videos')
parser.add_argument('-gb', '--generate-boat-images', action='store_true', help='Generate framed images for boats')
parser.add_argument('-gi', '--generate-images', action='store_true', help='Generate framed images')
parser.add_argument('-gtc', '--generate-transform-csv', action='store_true', help='Generate CSV with "video_id,x,y,angle"')

parser.add_argument('-uba', '--use-boat-angles', action='store_true', help='Use boat angles from ' + BOAT_ANGLES_CSV)

parser.add_argument('-cs', '--crop-size', type=int, default = 384, help='Crop size')
parser.add_argument('-cm', '--crop-margin', type=int, default = 64, help='Crop margin (really half of this on each side)')

parser.add_argument('-ati', '--analyze-train-images', action='store_true', help='Analyze mean/std of train images')
parser.add_argument('-abi', '--analyze-boat-images', action='store_true', help='Analyze mean/std of boat images')

args = parser.parse_args()

CROP_SIZE   = args.crop_size
CROP_MARGIN = args.crop_margin  # it's really CROP_MARGIN/2 on each side

assert args.generate_images or args.generate_video or args.generate_transform_csv or args.generate_boat_images or args.analyze_train_images or args.analyze_boat_images

def angle_diff(x, y):
    return np.arctan2(np.sin(x-y), np.cos(x-y))

def remove_orientation(yaw):
    yaw = np.fmod(yaw, 180.)
    if yaw >= 90.:
        yaw -= 180.
    elif yaw <= -90.:
        yaw += 180.
    assert (yaw <= (90.)) and (yaw >= (-90.))
    return yaw

def rotate(a, angle):
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta),  np.cos(theta)]])
    return np.dot(a, rotMatrix)

frame = pd.read_csv('training.csv')

if args.generate_transform_csv:
    transform_frame = pd.DataFrame(columns=['video_id', 'x', 'y', 'angle', 'dx', 'dy'])
    transform_frame.index.name = 'row_id'

if args.generate_images or args.generate_video or args.generate_transform_csv or args.generate_boat_images:
    if args.use_boat_angles:
        boat_angles_frame = pd.read_csv(BOAT_ANGLES_CSV).set_index('video_id')  

    for it, (video_id, video_frame) in enumerate(tqdm(frame.groupby(['video_id']))):
        # estimate mean of head and tail and termine based on variance which one is head (less variance)
        p1 = video_frame.dropna()[['x1', 'y1']]
        p2 = video_frame.dropna()[['x2', 'y2']]

        p1_mse = np.linalg.norm(p1 - p1.mean(), axis=1)
        p2_mse = np.linalg.norm(p2 - p2.mean(), axis=1)

        head_first = True
        if np.amax(p1_mse) > np.amax(p2_mse):
            head, tail = p2, p1
            head_first = False
        else:
            head, tail = p1, p2

        head.columns = tail.columns = ['x', 'y']
        cx,cy = (head.mean() + tail.mean()) / 2.
        vx,vy = (tail-head).mean()
        if args.use_boat_angles: # Use boat angles from csv
            angle = boat_angles_frame.get_value(video_id, 'angle')
            dx = cx - boat_angles_frame.get_value(video_id, 'cx')
            dy = cy - boat_angles_frame.get_value(video_id, 'cy')
        else:
            angle  = remove_orientation(np.arctan2(vy,vx) * 180. / np.pi)
            dx = dy = 0

        angles = np.arctan2((tail-head)['y'], (tail-head)['x']) * 180. / np.pi
        angle_error = np.absolute(np.amax(angle_diff(angles,angle))) # for debugging

        # average center of fish

        if args.generate_transform_csv:
            transform_frame.loc[it] = [video_id, cx,cy,angle,dx,dy ]

        if args.generate_images or args.generate_video or args.generate_boat_images:

            p0 = np.array((cx,cy))
            p1 = p0 + rotate(np.array([(CROP_SIZE-CROP_MARGIN)/2, 0.  ]), -angle) 
            p2 = p1 + rotate(np.array([0. ,(CROP_SIZE-CROP_MARGIN)/2]), -angle)

            dp = np.float32(([CROP_SIZE/2,CROP_SIZE/2], [CROP_SIZE - CROP_MARGIN/2,CROP_SIZE/2], [CROP_SIZE - CROP_MARGIN/2,CROP_SIZE - CROP_MARGIN/2])) 

            M = cv2.getAffineTransform(np.float32((p0,p1,p2)), dp)

            video_in  = cv2.VideoCapture(join(TRAIN_FOLDER, video_id) + '.mp4')
            fourcc    = cv2.VideoWriter_fourcc(*'MJPG')
            if args.generate_video:
                video_out = cv2.VideoWriter(join(TRAIN_CROPS_FOLDER, video_id)  + '-' + str(head_first) + '_' +str(int(angle)) + '.avi',fourcc, 15., (CROP_SIZE,CROP_SIZE))
            rot_video = np.empty([CROP_SIZE,CROP_SIZE,3], dtype=np.float32)
            it = 0
            boat_frame_id = 0
            BOAT_FRAMES = 3
            boat_frames = np.empty((BOAT_FRAMES,720, 1280,3), dtype=np.uint8)
            while(video_in.isOpened()):
                ret, frame = video_in.read()
                frame_id = video_frame['frame']
                if ret==True:
                    rot_video = cv2.warpAffine(frame,M,(CROP_SIZE,CROP_SIZE))
                    if args.generate_video: video_out.write(rot_video)
                    if (frame_id == it).any() and args.generate_images:
                        cv2.imwrite(join(TRAIN_IMAGES_FOLDER, video_id)  + '_' + str(it) + '.png', rot_video)
                    if args.generate_boat_images:
                        frame_id_nonan = video_frame.dropna()['frame']
                        if ((frame_id_nonan + boat_frame_id) == it).any():
                            boat_frames[boat_frame_id,:] = frame
                            boat_frame_id += 1
                            if boat_frame_id == BOAT_FRAMES:
                                for boat_it in range(BOAT_FRAMES):
                                    suffix = '_f' if boat_it == 0 else ''
                                    cv2.imwrite(join(TRAIN_BOATS_FOLDER, video_id)  + '_' + str(it-BOAT_FRAMES+1+boat_it) + suffix + '.png', boat_frames[boat_it])
                                boat_frame_id = 0

                else:
                    break
                it += 1
            video_in.release()
            if args.generate_video:
                video_out.release()

    if args.generate_transform_csv:
        from sklearn.cluster import DBSCAN
        X = np.dstack((
            transform_frame['x'].values, 
            transform_frame['y'].values,
            np.fmod(transform_frame['angle'].values + 15.,90.)/10. # do this so -89 and +89 fall close (see notebook)
            )).squeeze()
        clustering = DBSCAN(min_samples=0, eps=20.).fit(X)
        labels = clustering.labels_
        n_clusters = len(set(labels))
        assert n_clusters == 5
        transform_frame['boat_id'] = labels
        angles = transform_frame['angle'].values
        
        boat_angles_frame = pd.DataFrame(columns=['video_id', 'boat_id', 'angle', 'cx', 'cy']).set_index('video_id')

        for video_id,row_df in transform_frame.groupby('video_id'):
            boat_id = int(row_df['boat_id'].values)
            boat_angles_frame.loc[video_id]=[
                boat_id, 
                (np.mod(angles[labels==boat_id] +90. + 15, 180)-90).mean()-15,
                transform_frame[labels==boat_id]['x'].values.mean(),
                transform_frame[labels==boat_id]['y'].values.mean(),
                ]

        #for label in unique_labels:
        #    boat_angles_frame.loc[label]=(np.mod(angles[labels==label] +90. + 15, 180)-90).mean()-15
        
        transform_frame.to_csv('training_transform.csv')
        boat_angles_frame.to_csv(BOAT_ANGLES_CSV)

def stat_moments(folder):
    means = []
    stds  = []
    for filename in tqdm(glob.glob(os.path.join(folder, '*.png'))[:]):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)[::-1]
        means.append(np.mean(image,(0,1)))
        stds.append(np.std(image,(0,1)))
    means = np.array(means)
    stds  = np.array(stds)
    mean = np.mean(means,0)
    std  = np.mean(stds,0)
    print("Mean/std (0-255):", mean, std)
    print("Mean/std (0-1):", mean/255., std/255.)

if args.analyze_train_images:
    stat_moments(TRAIN_IMAGES_FOLDER)

if args.analyze_boat_images:
    stat_moments(TRAIN_BOATS_FOLDER)



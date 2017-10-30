# manually engineered way of generating fish sequences using probabilities on a per-frame basis
# this has a LAST DAY plan b/c I couldn't make the RNN approach to work.
# Scores 0.587 on the leaderboard.

import pandas as pd
import numpy as np
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--in-csv0', default='test_crops.csv', help='Load model from file')
parser.add_argument('-f1', '--in-csv1', default='test_crops_fishnet_046_0.0009.csv', type=str, help='Batch size')
parser.add_argument('-o', '--out-csv',  default='test_crops_submission.csv', help='Suffix to store checkpoints')
parser.add_argument('-l', '--len',  default=3, type=int, help='Suffix to store checkpoints')
parser.add_argument('-g', '--gap',  default=10, type=int, help='Suffix to store checkpoints')

args = parser.parse_args()

test_frame_X = pd.read_csv(args.in_csv0).set_index('video_id')
test_frame_X0 = pd.read_csv(args.in_csv1).set_index('video_id')

submission_frame = pd.DataFrame(test_frame_X)

fish_species = [u'species_fourspot', u'species_grey sole', u'species_other', u'species_plaice', u'species_summer', u'species_windowpane', u'species_winter']
all_species = fish_species + [u'species_no_fish']

def remove_species_no_fish(probs, threshold=0.98):
    fish_probs = np.expand_dims(np.sum(probs[probs[:,7] <= threshold,:7], axis=1), axis=1)
    probs[probs[:,7] <= threshold, :7] /= fish_probs

    # remove probs when no fish
    probs[probs[:,7] > threshold, :7] = 0 #np.nan 
    return probs

def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

for vid_id, vf in tqdm(test_frame_X.groupby('video_id')):

    f = test_frame_X.loc[vid_id]['species_no_fish'].values
    c = test_frame_X.loc[vid_id][[u'species_fourspot', u'species_grey sole',
           u'species_other', u'species_plaice', u'species_summer',
           u'species_windowpane', u'species_winter', 'species_no_fish']].values
    cs = np.argmax(c, axis=1)

    f0 = test_frame_X0.loc[vid_id]['species_no_fish'].values
    c0 = test_frame_X0.loc[vid_id][[u'species_fourspot', u'species_grey sole',
           u'species_other', u'species_plaice', u'species_summer',
           u'species_windowpane', u'species_winter', 'species_no_fish']].values
    cs0 = np.argmax(c0, axis=1)

    ff = np.bitwise_and((f < 0.01), (f0 < 0.01))

    l, s, v = rle(ff)

    MIN_LEN = args.len
    MIN_GAP = args.gap

    offsets = s[v & (l >= MIN_LEN)]
    if offsets[0] != 0:
        offsets = np.hstack(([0], offsets))
    assert offsets[0] == 0
    gaps = np.diff(offsets)
    filtered_offsets = np.hstack((offsets[0], offsets[1:][gaps > MIN_GAP]))
    frame_offsets = filtered_offsets

    fish_frames = []
    for frame_offset in frame_offsets:
        min_len = MIN_LEN if frame_offset != 0 else 1
        offset_range = range(frame_offset,frame_offset+min_len)
        species = (cs[offset_range])
        if True: #np.unique(species).size <= 1:
    #        print(c[offset_range,species])
            frame_align_max_pred = (np.argmax(c[offset_range, species]))
            fish_frames.append(frame_offset+frame_align_max_pred)
        else:
            print("FILTERED", offset_range)
            print(c[offset_range])

    fish_number = np.zeros(submission_frame.loc[vid_id]['frame'].size, dtype=np.float32)
    current_fish = 0
    for it in range(fish_number.size):
        if it in fish_frames:
            current_fish += 1
        fish_number[it] = current_fish

    submission_frame.loc[vid_id,'fish_number']=fish_number
    submission_frame.loc[vid_id]
    #print(fish_frames, len(fish_frames))

test_mismatch = [
    ('bc6hkwua3iLReunk', 1),
    ('8jkQWJWPCtIvcnmH', 1),
 #   ('tJinkrdMMZ477RGi', -3),
    ('P3QkoeOjxoM6pDKb', 172),
    ('pGd0FSJQcDH5DI8x', 1),
    ('Sw0AgnH8BY1BDGHu', 1),
    ('ZU6XtvFk0UMrHLEL', 1),
    ('LU2DSX6VZcIsiyaW', 1)]

submission_frame[all_species]=remove_species_no_fish(test_frame_X[all_species].values)
submission_frame = submission_frame.drop(u'species_no_fish', axis=1)

submission_frame = submission_frame.reset_index().drop('row_id', 1).set_index(['video_id', 'frame'])

# remove some frames
submission_frame = submission_frame.drop([('tJinkrdMMZ477RGi', 426), ('tJinkrdMMZ477RGi', 427), ('tJinkrdMMZ477RGi', 428)])
for m_vid, m_frame in test_mismatch:
    last_frame =len(submission_frame.loc[m_vid])
    for frame_to_add in range(last_frame, last_frame+m_frame):
        submission_frame.loc[(m_vid, frame_to_add), :] = 0

submission_frame=submission_frame.reset_index()
submission_frame=submission_frame[['frame','video_id', 'fish_number','length','species_fourspot','species_grey sole','species_other','species_plaice','species_summer','species_windowpane','species_winter']]
submission_frame.index.name='row_id'

submission_frame.to_csv(args.out_csv)
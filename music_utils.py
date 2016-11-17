'''
Presents general utility functions for 
music data processing from the million
song dataset
'''

import sklearn
from hdf5_getters import *
import os

##### Global constants #####
kNumPitches = 12
##### End global constants #####

def extract_data(filename):
    '''
    Reads data from filename, extracts relevant
    info and places it in a dict which is returned
    Builds off of abstractions provided in 
    hdf5_getters.py
    documentation: http://labrosa.ee.columbia.edu/millionsong/pages/example-track-description
    '''
    h5 = open_h5_file_read(filename)
    info = {}
    info['title'] = get_title(h5) # defaults to first song
    info['tempo'] = get_tempo(h5)
    info['artist'] = get_artist_name(h5)
    info['pitches'] = get_segments_pitches(h5)
    info['terms'] = get_artist_terms(h5)
    info['hotness'] = get_song_hotttnesss(h5)
    info['beats'] = get_beats_start(h5)
    info['bars'] = get_bars_start(h5)
    info['segments'] = get_segments_start(h5)
    h5.close()
    return info

def data_to_words(info, mode = 'perms'):
    '''
    Reads the data dict returned by extract_data
    and converts it to a language format
    '''
    if mode is 'perms':
        return pitch_perms(info) # pitch permutations
    else:
        raise Exception('Requested mode \"{mode}\" is not supported'.format(mode=mode))

def pitch_perms(info):
    '''
    Returns list of permutations of pitch indices
    corresponding to weights in pitches 
    array in decreasing order.
    Returns list of permutations of np.arangeS
    '''
    perms = []
    for pitch in info['pitches']:
        perm = sorted(np.arange(kNumPitches), key = lambda i: -pitch[i])
        perms.append(perm)
    return perms


def generate_data(data_path):
    '''
    Return a list of the info dicts returned
    by extract_data
    data_path: path to the data directory
    '''
    dataset = []
    for root, dirs, files in os.walk(data_path):
        files = glob.glob(os.path.join(root, '*h5'))
        for filename in files:
            dataset.append(extract_data(filename))
    return dataset

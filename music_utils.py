'''
Presents general utility functions for 
music data processing from the million
song dataset
'''

# import sklearn
#from hdf5_getters import *
import os
import numpy as np
import glob 
from collections import Counter



##### Global constants #####
kNumPitches = 12
kNoteFrequencies = [261.626, 277.183, 293.665, 311.127, 329.628, 349.228,\
 369.994, 391.995, 415.305, 440.000, 466.164, 493.883]
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
    #info['title'] = get_title(h5) # defaults to first song
    info['tempo'] = get_tempo(h5)
    #info['artist'] = get_artist_name(h5)
    info['pitches'] = get_segments_pitches(h5)
    #info['terms'] = get_artist_terms(h5)
    info['genre'] = get_artist_mbtags(h5)
    #info['hotness'] = get_song_hotttnesss(h5)
    #info['beats'] = get_beats_start(h5)
    #info['bars'] = get_bars_start(h5)
    info['segments'] = get_segments_start(h5)
    info['time signature'] = get_time_signature(h5)
    info['loudness'] = get_segments_loudness_max(h5)
    info['mode'] = get_mode(h5)
    info['key'] = get_key(h5)
    info['filename'] = filename # for debugging 
    h5.close()
    return info

def data_to_words(info, mode = 'perms'):
    '''
    Reads the data dict returned by extract_data
    and converts it to a language format
    '''
    if mode is 'perms':
        return pitch_perms(info) # pitch permutations
    elif mode is 'discretized':
        return discrete_segments(info)
    elif mode is 'max pitches':
        return 
    elif mode is 'max pitch diffs':
        return 
    else:
        raise Exception('Requested mode \"{mode}\" is not supported'.format(mode=mode))

def max_pitches(info):
    return np.array(map(np.argmax, info[pitches]))

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

def diffs(arr):
    '''
    Find diffs between adjacent entries 
    '''
    return np.array([arr[i + 1] - arr[i] for i in xrange(arr.shape[0] - 1)])

def discrete_segments(info, mult = 100):
    '''
    Takes segment beginnings representation
    and returns discretized version (number
    of occurences is proportional to interval
    length). Note we only look at all but the
    last segment (no interval for last segment)
    '''
    time_intervals = diffs(info['segments']) * mult
    return reduce(lambda x,y: x + y, map(lambda i: [i] * int(round(time_intervals[i])), range(len(time_intervals))))


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

### SONG TO WORD REPRESENTATIONS
# in first pass implementation, converts each segment to max pitch
def convert_raw_to_word(h5):
    return np.argmax(h5,axis=1)

def make_one_hot(idx):
    one_hot = np.zeros(4, dtype=np.int)
    one_hot[idx] = 1
    return one_hot

### FEATURE EXTRACTORS
def bag_of_words(song):
    init_dict = Counter({i: 0 for i in range(12)})
    init_dict.update(Counter([song[i] for i in range(song.size)]))
    bg_freq = init_dict
    freqs = bg_freq.values()
    return freqs

def bigrams(song):
    pitches = np.argmax(song['pitches'],axis=1)
    init_dict = Counter({(i,j): 0 for i in range(12) for j in range(12)})
    init_dict.update(Counter([(pitches[i-1],pitches[i]) for i in range(1,pitches.size)]))
    bg_freq = init_dict
    freqs = bg_freq.values()
    return freqs

def trigrams(song):
    pitches = np.argmax(song['pitches'],axis=1)
    init_dict = Counter({(i,j,k): 0 for i in range(12) for j in range(12) for k in range(12)})
    init_dict.update(Counter([(pitches[i-2],pitches[i-1],pitches[i]) for i in range(2,pitches.size)]))
    p_tg_freq = init_dict
    loudness = np.round(song['loudness'], decimals=0)
    init_dict = Counter({(i,j,k): 0 for i in range(10) for j in range(10) for k in range(10)})
    init_dict.update(Counter([(loudness[i-2],loudness[i-1],loudness[i]) for i in range(2,pitches.size)]))
    l_tg_freq = init_dict
    return p_tg_freq.values() + l_tg_freq.values()

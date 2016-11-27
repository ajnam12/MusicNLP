'''
Presents general utility functions for 
music data processing from the million
song dataset
'''

# import sklearn
from hdf5_getters import *
import os
import numpy as np
import glob 
from collections import Counter
from pyo import *
from time import sleep

##### Global constants #####
kNumPitches = 12
kNoteFrequencies = [261.626, 277.183, 293.665, 311.127, 329.628, 349.228,\
 369.994, 391.995, 415.305, 440.000, 466.164, 493.883]
##### End global constants #####

s = Server()
s.boot()

def play_song(pitches_arrays, segment_starts):
    lengths = segments_lengths(segment_starts)
    sine = Sine()
    sine.out()
    s.start()
    for i in xrange(len(pitches_arrays) - 1): # can't get the last one
        freq = kNoteFrequencies[np.argmax(pitches_arrays[i])]
        sine.freq = freq
        sleep(lengths[i])
    s.stop()

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

def segments_lengths(segments):
    '''
    Takes segments beginnings array
    finds differences between adjacent
    entries
    '''
    return np.array([segments[i + 1] - segments[i] for i in xrange(segments.shape[0] - 1)])

def discrete_segments(info, mult = 100):
    '''
    Takes segment beginnings representation
    and returns discretized version (number
    of occurences is proportional to interval
    length). Note we only look at all but the
    last segment (no interval for last segment)
    '''
    time_intervals = segments_lengths(info['segments']) * mult
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
    init_dict = Counter({(i,j): 0 for i in range(12) for j in range(12)})
    init_dict.update(Counter([(song[i-1],song[i]) for i in range(1,song.size)]))
    bg_freq = init_dict
    freqs = bg_freq.values()
    return freqs

def trigrams(song):
    init_dict = Counter({(i,j,k): 0 for i in range(12) for j in range(12) for k in range(12)})
    init_dict.update(Counter([(song[i-2],song[i-1],song[i]) for i in range(2,song.size)]))
    tg_freq = init_dict
    freqs = tg_freq.values()
    return freqs
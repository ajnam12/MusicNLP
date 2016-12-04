from pyo import *
from music_utils import *
import numpy as np
from time import sleep
# convolutions w/ discretization
# tfidx for co-occurences of pitch sequences (or discretized)

s = Server()
s.boot()

def play_song(pitches_arrays, segment_starts, server=None):
    lengths = diffs(segment_starts)
    sine = Sine()
    sine.out()
    if server == None:
        server = s
    server.start()
    for i in xrange(len(pitches_arrays) - 1): # can't get the last one
        freq = kNoteFrequencies[np.argmax(pitches_arrays[i])]
        #freq = kNoteFrequencies
        sine.freq = freq
        #sine.mul = list(pitches_arrays[i])
        sleep(lengths[i])
    server.stop()
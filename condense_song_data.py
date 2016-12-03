from hdf5_getters import *
from music_utils import *
import os
import numpy as np
import glob
import copy
import time
from multiprocessing import Process, Pool
from collections import Counter
import cPickle as pickle

print "top"


def condenseLetters(letter):
    print "Launching process " + letter
    genres = ['rock', 'jazz', 'folk', 'hip hop rnb and dance hall']
    genres2idx = dict(zip(genres, range(len(genres))))
    data_root = '/mnt/snap/'
    data_path = data_root + 'data/' + letter
    genre_dict = {genre: [] for genre in genres}
    counter = 0
    for root, dirs, files in os.walk(data_path):
        files = glob.glob(os.path.join(root, '*h5'))
        for f in files:
            song = extract_data(f)
            song_cpy = copy.copy(song) 
            terms = song['genre']
            intersection = list(set(terms) & set(genres))
            if len(intersection) == 1:
                del song_cpy['genre']
                genre_dict[intersection[0]].append(song_cpy)
        if len(dirs) > 0:
            print root
            counter += 1
            print "%s: Finished processing %d/27\n" % (letter, counter)
    f = open(data_root + "genre_dicts/" + letter + "_genre_dict.pkl", "wb")
    pickle.dump(genre_dict, f) 
    f.close()
    print "Process " + letter + " complete!"

def condenseGenres():
    genres = ['rock', 'jazz', 'folk', 'hip hop rnb and dance hall']
    genres2idx = dict(zip(genres, range(len(genres))))
    data_root = '/mnt/snap/genre_dicts'
    name = 'full_genre_dict.pkl'
    genre_dict = {genre: [] for genre in genres}
    for root, dirs, files in os.walk(data_root):
        files = glob.glob(os.path.join(root, '*.pkl'))
        for f in files:
            print "Processing %s..." % f
            f_desc = open(f, 'rb')
            d = pickle.load(f_desc)
            f_desc.close()
            for k,v in d.iteritems():
                genre_dict[k] += v
    f = open(data_root + '/' + name, 'wb')
    pickle.dump(genre_dict, f)
    f.close()

#alphabet = list("DEFGHJKLMNOPQRTVYZ")
#alphabet = list("A")
#alphabet = ["%s/%s"%(a,b) for a in alphabet for b in alphabet]
#pool = Pool(processes=min(len(alphabet), 30))
#pool.map(condenseLetters, alphabet)

condenseGenres()
#condenseLetters('S')
print "bottom"

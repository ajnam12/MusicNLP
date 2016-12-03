import glob
from hdf5_getters import *
import os
import numpy as np
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2, l1
from music_utils import *
import cPickle as pickle



### NN MODELS
def vanilla_model(input_dim, output_dim, hidden_dim=50, num_layers=1, reg=0.05):
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=input_dim, W_regularizer=l2(reg), init='glorot_normal'))
    model.add(Activation('tanh'))
    for i in range(1,num_layers):
      model.add(Dense(hidden_dim, W_regularizer=l2(reg), init='glorot_normal'))
      model.add(Activation('tanh'))
    model.add(Dense(output_dim, W_regularizer=l2(reg), init='glorot_normal'))
    model.add(Activation('softmax'))
    return model

def make_train_example(h5, feature_extractor):
    song = convert_raw_to_word(h5)
    return feature_extractor(song)
    #pitches = get_segments_pitches(h5)[:11] # limit: only look at beginning
    #pitch_diffs = [pitches[i] - pitches[i - 1] for i in xrange(1, len(pitches))]
    #return {'title': title, 'pitch_diffs': pitch_diffs}

         
### MAKE DATASET
"""
genres = ['rock', 'punk', 'folk', 'hip hop rnb and dance hall']
genre_idxs = dict(zip(genres, range(len(genres))))
genre_songs = {'rock': [], 'punk': [], 'folk': [], 'hip hop rnb and dance hall': []}

tags_list = []
data_path = "~/MillionSongSubset/data"
for root, dirs, files in os.walk("MillionSongSubset/data"):
    files = glob.glob(os.path.join(root, '*h5'))
    for f in files:
        h5 = open_h5_file_read(f)
        tags = get_artist_mbtags(h5).tolist()
        tags_list += tags
        for tag in tags:
          if tag in genre_songs:
            genre_songs[tag].append(make_train_example(get_segments_pitches(h5),fe))
            #h5.close()
            #exit()
            break
        h5.close()
        #train_pair = make_train_pair(h5)
        #titles.append(train_pair['title'])
        #pitch_diff_list.append(train_pair['pitch_diffs'])
"""


def load_dataset():
    with open('../train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('../test.p', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, test_data

def make_dataset(train_data, test_data, feature_extractor, genres, genre_idxs):
    #train_data, test_data = load_dataset()
    print "data loaded"
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    print "extracting train data"
    for k,v in train_data.iteritems():
        print "- extracting %s"%k
        y_train += [make_one_hot(genre_idxs[k])]*len(v)
        for song in v:
            feats = feature_extractor(song)
            print len(feats)
            x_train.append(feats)
            exit()
    print "extracting test data"
    for k,v in test_data.iteritems():
        print "- extracting %s"%k
        y_test += [make_one_hot(genre_idxs[k])]*len(v)
        for song in v:
            feats = feature_extractor(song)
            x_test.append(feats)
    print "features extracted"
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print x_train.shape
    print y_train.shape
    x_train = np.hstack((x_train,y_train))
    exit()
    np.random.shuffle(x_train)
    x_train, y_train = np.hsplit(x_train, [-4])
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    return x_train, y_train, x_test, y_test

### TRAIN MODEL
def train_model(train_data, test_data):
    genres = ['jazz', 'hip hop rnb and dance hall', 'folk', 'rock']
    genre_idxs = dict(zip(genres, range(len(genres))))
    fe = trigrams
    x_train, y_train, x_test, y_test = make_dataset(train_data, test_data, fe, genres, genre_idxs)
    vec_size = 12**3 + 10**3
    regs = [0.005, 0.01, 0.05]
    lrs = [0.01, 0.05, 0.1]
    num_layers = [1, 2, 3]
    model = vanilla_model(vec_size, len(genres), num_layers=1, reg=0.01)
    sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    print "Training..."
    model.fit(x_train, y_train, nb_epoch=20,batch_size=50)
    print "Testing..."
    score = model.evaluate(x_test,y_test,batch_size=1)
    preds = model.predict(x_test,batch_size=1,verbose=1)
    print score


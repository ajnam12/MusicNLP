import glob
from hdf5_getters import *
import os
import numpy as np
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Nadam
from keras.regularizers import l2, l1
import music_utils
from music_utils import *
from graph_utils import *
import cPickle as pickle
import random
import sklearn


random.seed(221)
genres = ['jazz', 'hip hop rnb and dance hall', 'folk', 'rock']
genre_idxs = dict(zip(genres, range(len(genres))))

### NN MODELS
def vanilla_model(input_dim, output_dim, hidden_dim=200, num_layers=1, reg=0.05, non_linearity='tanh'):
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=input_dim, W_regularizer=l2(reg), init='glorot_normal'))
    model.add(Activation(non_linearity))
    model.add(Dropout(.5))
    for i in range(1,num_layers):
      model.add(Dense(hidden_dim, W_regularizer=l2(reg), init='glorot_normal'))
      model.add(Activation(non_linearity))
      model.add(Dropout(.5))
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


def load_dataset(from_full_dict=True):
    train_data = {}
    test_data = {}
    if not from_full_dict:
      with open('../train.p', 'rb') as f:
          train_data = pickle.load(f)
      with open('../test.p', 'rb') as f:
          test_data = pickle.load(f)
    else:
      with open('10k_genre_dict.pkl', 'rb') as f:
        data = pickle.load(f)
        for k,v in data.iteritems():
          test_data[k] = v[:1000]
          train_data[k] = v[1000:]
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
            x_train.append(feats)
    print "extracting test data"
    for k,v in test_data.iteritems():
        print "- extracting %s"%k
        y_test += [make_one_hot(genre_idxs[k])]*len(v)
        for song in v:
            feats = feature_extractor(song)
            x_test.append(feats)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.hstack((x_train,y_train))
    np.random.shuffle(x_train)
    x_train, y_train = np.hsplit(x_train, [-4])
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    return x_train, y_train, x_test, y_test

# genres = ['jazz', 'hip hop rnb and dance hall', 'folk', 'rock']
# genre_idxs = dict(zip(genres, range(len(genres))))
# fe = trigrams
# x_train, y_train, x_test, y_test = make_dataset(train_data, test_data, fe, genres, genre_idxs)
# vec_size = 12**3 + 21**2

### TRAIN MODEL
#def train_model(train_data, test_data):
def train_model(x_train, y_train, x_test, y_test, vec_size):
    genres = ['jazz', 'hip hop rnb and dance hall', 'folk', 'rock']
    genre_idxs = dict(zip(genres, range(len(genres))))
    
    print x_train.shape
    print x_test.shape
    regs = [0.005, 0.01, 0.05]
    lrs = [0.01, 0.05, 0.1]
    num_layers = [1, 2, 3]
    results = {}
    idx = 0
    tot = len(regs)*len(lrs)*len(num_layers)
    for num_layer in num_layers:
      for lr in lrs:
        for reg in regs:
          print "Combination %d/%d" % (idx, tot)
          model = vanilla_model(vec_size, len(genres), num_layers=num_layer, reg=reg, non_linearity='relu')
          opt = Adagrad()
          model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
          print "Training..."
          model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=40,batch_size=50)
          #model.fit(x_train, y_train, nb_epoch=10,batch_size=50)
          print "Testing..."
          score = model.evaluate(x_test,y_test,batch_size=1)
          preds = model.predict(x_test,batch_size=1,verbose=1)
          results[(num_layer, lr, reg)] = (score, preds)
          #print score
          idx += 1
    return (y_test, results)

### GRAPH RESULTS
def create_confusion_matrix(ground_truths, predictions):
  gts = np.argmax(ground_truths, axis=1)
  preds = np.argmax(predictions, axis=1)
  print ground_truths.shape[0]
  print gts.shape[0]
  print predictions.shape[0]
  print preds.shape[0]
  conf_mat = sklearn.metrics.confusion_matrix(gts, preds) 
  plot_confusion_matrix(conf_mat, genres, 'test')
  print 'bottom'

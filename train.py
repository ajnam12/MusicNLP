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
    #h5 = open_h5_file_read(filename)
    song = convert_raw_to_word(h5)
    #h5.close()
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
def make_dataset(feature_extractor):
    
    data_path = ""

    num_train = 175
    num_test = 25
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for k,v in genre_songs.iteritems():
      y_train += [make_one_hot(genre_idxs[k])]*num_train
      x_train += v[:num_train]
      y_test += [make_one_hot(genre_idxs[k])]*num_test
      x_test += v[-num_test:]

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

### TRAIN MODEL
fe = trigrams
vec_size = 12**3
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

#print Counter(tags_list).most_common(30)
#for k,v in genre_songs.iteritems():
#  print k + " " + str(len(v))

import glob
from hdf5_getters import *
import os
import numpy as np
from collections import Counter
from music_utils import *

tags_list = []
data_path = "/mnt/snap/data/"
count = 0
for root, dirs, files in os.walk(data_path):
    files = glob.glob(os.path.join(root, '*h5'))
    #if count > 1000: break 
    for f in files: 
      h5 = open_h5_file_read(f)
      tags = get_artist_mbtags(h5).tolist()
      tags_list += tags
      #count += 1
      h5.close()

print Counter(tags_list).most_common(100)

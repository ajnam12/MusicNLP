import matplotlib.pyplot as plt
import numpy as np
import sklearn
import itertools

def plot_confusion_matrix(conf_mat, classes, title, normalize=False):
  plt.figure()
  plt.imshow(conf_mat, interpolation='nearest')
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange((len(classes)))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    conf_mat = conf_mat.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    print "Normalized confusion matrix"
  else:
    print "Confusion matrix, without normalization"

  print conf_mat

  thresh = conf_mat.max() / 2.0
  for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
    plt.text(j, i, conf_mat[i, j], horizontalalignment="center", color="white" if conf_mat[i,j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

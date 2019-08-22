import numpy as np

from keras.utils import Sequence

class DataGenerator(Sequence):
  def __init__(self, X, Y, batch_size):
    self.X = X
    self.Y = Y
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.X) / float(self.batch_size)))

  def __getitem__(self, idx):
    X_batch = np.array(self.X[idx * self.batch_size:(idx + 1) * self.batch_size])
    Y_batch = np.array(self.Y[idx * self.batch_size:(idx + 1) * self.batch_size])
    
    X_msl = np.max([len(x) for x in X_batch])

    X_batch_new = list()
    for x in X_batch:
      x_new = list(x)
      x_new.extend([[0] * len(x[0])] * (X_msl - len(x_new)))
      X_batch_new.append(np.array(x_new))
    X_batch = X_batch_new

    return np.array(X_batch), np.array(Y_batch)
    
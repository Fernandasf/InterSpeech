import os
import sys
import numpy as np
import gc
from tqdm import tqdm

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.callbacks import Callback
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten

from validate import Validation
from dataloader import DataLoader

K.clear_session()

class ValidationCallback(Callback):
    def __init__(self, Batch_dev, data_folder, lab_dict, wav_lst_te, wlen, wshift, class_lay):
        self.wav_lst_te = wav_lst_te
        #self.data_folder = data_folder
        #self.wlen = wlen
        #self.wshift = wshift
        self.lab_dict = lab_dict
        self.batch_size = config.optimization.batch_size
        self.class_lay = class_lay
        input_list_tr = config.dataset.list_tr

    def on_epoch_end(self, epoch, logs={}):
        val = Validation(self.Batch_dev, self.data_folder, self.lab_dict, self.wav_lst_te, self.wlen, self.wshift,
                         self.class_lay, self.model)
        val.validate(epoch)

class Train():
    def __init__(self, config):
        dataloader = DataLoader(config)
        self.wlen, self.wshift = dataloader.shift_samples()
        self.fs = config.windowing.fs






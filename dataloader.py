import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
#import torch
#from torch.autograd import Variable

from util import read_txt

from keras.callbacks import Callback

class DataLoader():
    def __init__(self, config):
        self.cw_len = config.windowing.cw_len
        self.cw_shift = config.windowing.cw_shift
        self.fs = config.windowing.fs
        self.batch_size = config.optimization.batch_size
        self.fact_amp = config.optimization.fact_amp

        self.wlen, self.wshift = self.shift_samples()
        self.label_encoder = LabelEncoder()

        input_list_tr = config.dataset.list_tr
        #input_list_te = config.dataset.list_te

        self.list_tr = self.get_train_dataset(input_list_tr)
        #self.list_te, self.label_te = self.get_dataset(input_list_te, test=True)


    #@staticmethod
    def shift_samples(self):
        # Converting context and shift in samples
        wlen = int(self.fs * self.cw_len / 1000.00)  # 6000 # 3000
        wshift = int(self.fs * self.cw_shift / 1000.00)  # 160 #
        print("wlen e wshift: ", wlen, wshift)
        return wlen, wshift

    def create_batches_rnd(self):
        ## Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
        sig_batch = np.zeros([self.batch_size, self.wlen])  # 128, 6000
        lab_batch = [''] * self.batch_size  # 128
        len_list_tr = len(self.list_tr)
        index_rand_tr = np.random.randint(len_list_tr, size=self.batch_size)  # 128
        amp_rand_tr = np.random.uniform(1.0 - self.fact_amp, 1 + self.fact_amp, self.batch_size)  # 128

        for i in range(self.batch_size):
            # select a random sentence from the list
            filename = self.list_tr[index_rand_tr[i]]
            signal = librosa.load(filename, sr=self.fs)[0]

            # accesing to a random chunk
            sig_len = signal.shape[0]
            sig_beg = np.random.randint(sig_len - self.wlen - 1)  # randint(0, signal_len-2*wlen-1)
            sig_end = sig_beg + self.wlen

            sig_batch[i, :] = signal[sig_beg:sig_end] * amp_rand_tr[i]
            lab_batch[i] = str(filename.split('/')[-2])

        lab_batch_n = self.label_encoder.fit_transform(np.array(lab_batch))

        # inp = Variable(torch.from_numpy(sig_batch).float().contiguous())
        # lab = Variable(torch.from_numpy(lab_batch_n).contiguous())
        # return inp, lab
        a, b = np.shape(sig_batch)
        sig_batch = sig_batch.reshape((a, b, 1))
        return sig_batch, np.array(lab_batch_n)


    def get_train_dataset(self, input_list):
        list_ = read_txt(input_list)
        len_list = len(list_)
        print("Len Train list: ", len_list)
        return list_

    def get_test_dataset(self, input_list):
        list_ = read_txt(input_list)
        len_list = len(list_)
        print("Len Test list: ", len_list)
        lab_batch = [str(str(list_[i]).split('/')[-2]) for i in range(len_list)]
        labels = self.label_encoder.fit_transform(np.array(lab_batch))
        return list_, labels



        

# # TODO: add multiprocess to load audio
# def load_audio(wav_list):
#     sr = 16000
#     labels = []
#     audios = []
#     for file in wav_list:
#         label = file.split("/")[-2]
#         audio = librosa.load(file, sr=sr)[0]
#         audios.append(audio)
#         labels.append(label)
#     return audios, labels





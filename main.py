import argparse
import json
import numpy as np
import os
import librosa
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataloader import get_dataset, read_txt
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN

from sklearn.preprocessing import LabelEncoder

"""
1 - dataloader
3 - model - arquitetura
4 - treinamento
5 - teste
"""

def create_batches_rnd(batch_size, wlen, len_list_tr, list_tr, fact_amp, fs):
    ## Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = np.zeros([batch_size, wlen])  # 128, 6000
    lab_batch = [''] * batch_size  # 128
    index_rand_tr = np.random.randint(len_list_tr, size=batch_size)  # 128
    amp_rand_tr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)  # 128

    for i in range(batch_size):
        # select a random sentence from the list
        filename = list_tr[index_rand_tr[i]]
        signal = librosa.load(filename, sr=fs)[0]

        # accesing to a random chunk
        sig_len = signal.shape[0]
        sig_beg = np.random.randint(sig_len - wlen - 1)  # randint(0, signal_len-2*wlen-1)
        sig_end = sig_beg + wlen

        sig_batch[i, :] = signal[sig_beg:sig_end] * amp_rand_tr[i]
        lab_batch[i] = str(filename.split('/')[-2])

    label_n = LabelEncoder()
    lab_batch_n = label_n.fit_transform(np.array(lab_batch))

    inp = Variable(torch.from_numpy(sig_batch).float().contiguous())
    lab = Variable(torch.from_numpy(lab_batch_n).contiguous())

    return inp, lab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        help='Load settings from file in json format.')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_json = json.load(f)

    config = DotMap(config_json)
    print("args: ", config)
    # list to test with head samples
    # input_list_tr = config.input_list_tr
    # input_list_te = config.input_list_te

    input_list_tr = config.dataset.list_tr
    input_list_te = config.dataset.list_te
    output_folder = config.dataset.output_folder

    # setting seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Folder creation
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)

    # ------------ Model -> Feature_extraction + classifier --------


    cw_len = config.windowing.cw_len
    cw_shift = config.windowing.cw_shift
    fs = config.windowing.fs

    # load dataset train, test
    # audios, labels = get_dataset(input_list)
    print("_" * 80)
    print("Loading dataset....")
    list_tr = read_txt(input_list_tr)
    len_list_tr = len(list_tr)
    print("Len Train list: ", len_list_tr)
    list_te = read_txt(input_list_te)
    len_list_te = len(list_te)
    print("Len Test list: ", len_list_te)
    lab_batch_te = [str(str(list_te[i]).split('/')[-2]) for i in range(len_list_te)]
    label_n = LabelEncoder()
    lab_batch_te = label_n.fit_transform(np.array(lab_batch_te))

    # Converting context and shift in samples
    wlen = int(fs * cw_len / 1000.00) #6000
    wshift = int(fs * cw_shift / 1000.00) #160

    # Batch_dev
    Batch_dev = 128
    print("_"*80)
    print("Loading Model")
    # loss function
    cost = nn.NLLLoss()

    # Feature extractor CNN
    CNN_arch = {'input_dim': wlen,
                'fs': fs,
                'cnn_N_filt': config.cnn_arch.cnn_N_filt,
                'cnn_len_filt': config.cnn_arch.cnn_len_filt,
                'cnn_max_pool_len': config.cnn_arch.cnn_max_pool_len,
                'cnn_use_laynorm_inp': config.cnn_arch.cnn_use_laynorm_inp,
                'cnn_use_batchnorm_inp': config.cnn_arch.cnn_use_batchnorm_inp,
                'cnn_use_laynorm': config.cnn_arch.cnn_use_laynorm,
                'cnn_use_batchnorm': config.cnn_arch.cnn_use_batchnorm,
                'cnn_act': config.cnn_arch.cnn_act,
                'cnn_drop': config.cnn_arch.cnn_drop,
                }

    CNN_net = CNN(CNN_arch)
    # CNN_net.cuda()
    # print(CNN_net)

    DNN1_arch = {'input_dim': CNN_net.out_dim,
                 'fc_lay': config.dnn_arch_1.fc_lay,
                 'fc_drop': config.dnn_arch_1.fc_drop,
                 'fc_use_batchnorm': config.dnn_arch_1.fc_use_batchnorm,
                 'fc_use_laynorm': config.dnn_arch_1.fc_use_laynorm,
                 'fc_use_laynorm_inp': config.dnn_arch_1.fc_use_laynorm_inp,
                 'fc_use_batchnorm_inp': config.dnn_arch_1.fc_use_batchnorm_inp,
                 'fc_act': config.dnn_arch_1.fc_act,
                 }

    DNN1_net = MLP(DNN1_arch)
    # DNN1_net.cuda()
    # print(DNN1_net)
    class_lay = config.dnn_arch_2.class_lay

    DNN2_arch = {'input_dim': config.dnn_arch_1.fc_lay[-1],
                 'fc_lay': config.dnn_arch_2.class_lay,
                 'fc_drop': config.dnn_arch_2.class_drop,
                 'fc_use_batchnorm': config.dnn_arch_2.class_use_batchnorm,
                 'fc_use_laynorm': config.dnn_arch_2.class_use_laynorm,
                 'fc_use_laynorm_inp': config.dnn_arch_2.class_use_laynorm_inp,
                 'fc_use_batchnorm_inp': config.dnn_arch_2.class_use_batchnorm_inp,
                 'fc_act': config.dnn_arch_2.class_act,
                 }

    DNN2_net = MLP(DNN2_arch)
    # DNN2_net.cuda()
    # print(DNN2_net)

    lr = config.optimization.lr

    optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
    optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
    optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

    # ---------------------- Trainer --------------------------
    print("_" * 80)
    print("Training and Test Model by epoch")
    batch_size = config.optimization.batch_size
    N_epochs = config.optimization.N_epochs
    N_batches = config.optimization.N_batches
    fact_amp = config.optimization.fact_amp
    N_eval_epoch = config.optimization.N_eval_epoch

    for epoch in range(N_epochs):

        test_flag = 0
        CNN_net.train()
        DNN1_net.train()
        DNN2_net.train()

        loss_sum = 0
        err_sum = 0

        for i in range(N_batches):
            [inp, lab] = create_batches_rnd(batch_size, wlen, len_list_tr, list_tr, fact_amp, fs)
            pout = DNN2_net(DNN1_net(CNN_net(inp)))

            pred = torch.max(pout, dim=1)[1]
            loss = cost(pout, lab.long())
            err = torch.mean((pred != lab.long()).float())

            optimizer_CNN.zero_grad()
            optimizer_DNN1.zero_grad()
            optimizer_DNN2.zero_grad()

            loss.backward()
            optimizer_CNN.step()
            optimizer_DNN1.step()
            optimizer_DNN2.step()

            loss_sum = loss_sum + loss.detach()
            err_sum = err_sum + err.detach()

        loss_tot = loss_sum / N_batches
        err_tot = err_sum / N_batches

        # import sys
        # sys.exit("stop code!")
        # ------------ Full Validation new ---------------
        if epoch % N_eval_epoch == 0:

            CNN_net.eval()
            DNN1_net.eval()
            DNN2_net.eval()
            test_flag = 1
            loss_sum = 0
            err_sum = 0
            err_sum_snt = 0

            with torch.no_grad():
                lab_batch = [''] * len_list_te
                for i in range(len_list_te):
                    filename = str(list_te[i])
                    signal = librosa.load(filename, sr=fs)[0]
                    signal = torch.from_numpy(signal).float().contiguous()
                    lab_batch = lab_batch_te[i]

                    # split signals into chunks
                    beg_samp = 0
                    end_samp = wlen

                    N_fr = int((signal.shape[0] - wlen) / (wshift)) #62
                    #print("N_fr: ", N_fr)
                    #print("lab_batch_n ", lab_batch)

                    sig_arr = torch.zeros([Batch_dev, wlen]).float().contiguous()
                    #print("sig_arr ", sig_arr)
                    lab = Variable((torch.zeros(N_fr + 1) + lab_batch).contiguous().long())
                    #print("lab ", lab)
                    pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().contiguous())
                    #print("pout ", pout)
                    count_fr = 0
                    count_fr_tot = 0
                    while end_samp < signal.shape[0]:
                        sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                        # print("len sig_arr ", len(sig_arr))
                        beg_samp = beg_samp + wshift
                        end_samp = beg_samp + wlen
                        count_fr = count_fr + 1
                        count_fr_tot = count_fr_tot + 1

                        if count_fr == Batch_dev:
                            inp = Variable(sig_arr)
                            pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                            count_fr = 0
                            sig_arr = torch.zeros([Batch_dev, wlen]).float().contiguous()

                    if count_fr > 0:
                        inp = Variable(sig_arr[0:count_fr])
                        pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))

                    pred = torch.max(pout, dim=1)[1]
                    loss = cost(pout, lab.long())
                    err = torch.mean((pred != lab.long()).float())

                    [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                    err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                    loss_sum = loss_sum + loss.detach()
                    err_sum = err_sum + err.detach()

                    err_tot_dev_snt = err_sum_snt / len_list_te
                    loss_tot_dev = loss_sum / len_list_te
                    err_tot_dev = err_sum / len_list_te

                print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

                with open(output_folder + "/res.res", "a") as res_file:
                    res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                    epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

                checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                              'DNN1_model_par': DNN1_net.state_dict(),
                              'DNN2_model_par': DNN2_net.state_dict(),
                              }
                torch.save(checkpoint, output_folder + '/model_raw.pkl')

        else:
            print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
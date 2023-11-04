import argparse
import json
import numpy as np
import os
import librosa
from dotmap import DotMap
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataloader import DataLoader
from models import MLP, Dense, SincNet


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

    output_folder = config.dataset.output_folder

    # setting seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Folder creation
    os.makedirs(output_folder, exist_ok=True)

    # ------------ Model -> Feature_extraction + classifier --------
    dataloader = DataLoader(config)
    wlen, wshift = dataloader.shift_samples()
    fs = config.windowing.fs

    input_list_te = config.dataset.list_te
    list_te, label_te = dataloader.get_test_dataset(input_list_te)

    # Batch_dev
    #Batch_dev = 128
    Batch_dev = config.batch_dev
    print("_"*80)
    print("Loading Model")
    # loss function
    cost = nn.NLLLoss()

    # Feature extractor CNN
    sinc_net = SincNet(config, wlen)
    # sinc_net.cuda()
    # pprint(sinc_net)

    mlp_net = MLP(config, sinc_net.out_dim)
    # mlp_net.cuda()
    # print(mlp_net)

    class_lay = config.dnn_arch_2.class_lay
    input_dim = config.dnn_arch_1.fc_lay[-1]
    dense_net = Dense(config, input_dim)
    # dense_net.cuda()
    # pprint(dense_net)

    lr = config.optimization.lr

    optimizer_CNN = optim.RMSprop(sinc_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
    optimizer_DNN1 = optim.RMSprop(mlp_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
    optimizer_DNN2 = optim.RMSprop(dense_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

    # ---------------------- Trainer --------------------------
    print("_" * 80)
    print("Training and Test Model by epoch")
    # batch_size = config.optimization.batch_size
    N_epochs = config.optimization.N_epochs
    N_batches = config.optimization.N_batches
    # fact_amp = config.optimization.fact_amp
    N_eval_epoch = config.optimization.N_eval_epoch

    for epoch in range(N_epochs):

        test_flag = 0
        sinc_net.train()
        mlp_net.train()
        dense_net.train()

        loss_sum = 0
        err_sum = 0

        for i in range(N_batches):
            # TODO: check the i in N_batches
            [inp, lab] = dataloader.create_batches_rnd()
            pout = dense_net(mlp_net(sinc_net(inp)))

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
            # print("Validation: ", epoch)

            sinc_net.eval()
            mlp_net.eval()
            dense_net.eval()
            test_flag = 1
            loss_sum = 0
            err_sum = 0
            err_sum_snt = 0
            stn_sum = 0

            with torch.no_grad():
                lab_batch = [''] * len(list_te)
                for i in range(len(list_te)):
                    filename = str(list_te[i])
                    signal = librosa.load(filename, sr=fs)[0]
                    signal = torch.from_numpy(signal).float().contiguous()
                    lab_batch = label_te[i]

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
                            pout[count_fr_tot - Batch_dev:count_fr_tot, :] = dense_net(mlp_net(sinc_net(inp)))
                            count_fr = 0
                            sig_arr = torch.zeros([Batch_dev, wlen]).float().contiguous()

                    if count_fr > 0:
                        inp = Variable(sig_arr[0:count_fr])
                        pout[count_fr_tot - count_fr:count_fr_tot, :] = dense_net(mlp_net(sinc_net(inp)))

                    pred = torch.max(pout, dim=1)[1]
                    print("len(pred): ", len(pred))
                    print("pred: ", pred)
                    loss = cost(pout, lab.long())
                    err = torch.mean((pred != lab.long()).float())
                    # print("err: ", err)

                    [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                    print("val: ", val, "best_class: ", best_class)
                    err_sum_snt = err_sum_snt + (best_class != lab[0]).float()
                    print("(best_class != lab[0]): ", (best_class != lab[0]).float())

                    loss_sum = loss_sum + loss.detach()
                    err_sum = err_sum + err.detach()

                    # calculate ACC
                    stn_sum += 1
                    temp_acc_stn = str(round(1 - (err_sum_snt.detach().numpy() / stn_sum), 4))
                    temp_acc = str(round(1 - (err_sum.detach().numpy() / stn_sum), 4))

                    # print("temp_acc_stn ", temp_acc_stn)
                    # print("temp_acc ", temp_acc)

                    err_tot_dev_snt = err_sum_snt / len(list_te)
                    loss_tot_dev = loss_sum / len(list_te)
                    err_tot_dev = err_sum / len(list_te)

                    # average accuracy
                    acc = 1 - (err_sum / len(list_te))
                    acc_snt = 1 - (err_sum_snt / len(list_te))

                # print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
                # epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))
                # with open(output_folder + "/res.res", "a") as res_file:
                #     res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                #     epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

                print('Epoch: {}, acc_te: {}, acc_te_snt: {}\n'.format(epoch, acc, acc_snt))
                with open(output_folder + "/res.res", "a") as res_file:
                    res_file.write("epoch %i, acc_te=%f acc_te_snt=%f\n" % (epoch, acc, acc_snt))

                checkpoint = {'CNN_model_par': sinc_net.state_dict(),
                              'DNN1_model_par': mlp_net.state_dict(),
                              'DNN2_model_par': dense_net.state_dict(),
                              }
                torch.save(checkpoint, output_folder + '/model_raw.pkl')

        else:
            print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))
            # print('acc_te: {}, acc_te_snt: {}\n'.format(acc, acc_snt))

import numpy as np


def read_txt(txt_list):
    with open(txt_list, 'r') as f:
        wav_list = [line.strip() for line in f]
    return wav_list
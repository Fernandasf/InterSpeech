import librosa

# class Dataloader(config):
#     def __init__(self):
#         self.input_list =
#         self.sample_rate =

# TODO: add multiprocess to load audio
def load_audio(wav_list):
    sr = 16000
    labels = []
    audios = []
    for file in wav_list:
        label = file.split("/")[-2]
        audio = librosa.load(file, sr=sr)[0]
        audios.append(audio)
        labels.append(label)
    return audios, labels

def read_txt(txt_list):
    with open(txt_list, 'r') as f:
        wav_list = [line.strip() for line in f]
    return wav_list

def get_dataset(input_list):
    wav_list = read_txt(input_list)
    audios, labels = load_audio(wav_list)
    return audios, labels

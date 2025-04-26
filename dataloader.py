import librosa
from torch.utils import data
import random
import numpy as np
import pandas as pd
import os
import soundfile as sf
from scipy import signal

TRAIN_CLEAN_PATH = "D:\\Users\\14979\\Desktop\\DNS5_16k\\train_clean"
TRAIN_NOISE_CSV_PATH = "D:\\Users\\Database\\train_noise_data_new.csv"
RIR_PATH = "D:\\Users\\14979\\Desktop\\RIR"  # npy格式保存
VALID_DATASET_PATH = "D:\\Users\\14979\\Desktop\\TGRU_dataset"


def add_reverb(clean, rir):
    # 下面这一条语句可用于对numpy中的一维数组扩展维度 从(n,)转为(1, n) 类似于torch中的unsqueeze操作
    # 因为输入的rir是(2, L)的 所以需要这步操作扩展信号的维度(否则会报错)
    clean = clean[None]  # (T,) -> (1, T)
    reverb = signal.fftconvolve(clean, rir, mode='full')  # (2, T_reverb)
    reverb = reverb[:, :clean.shape[-1]]

    return reverb


def mk_mixture(speech, noise, rir, snr, eps=1e-8):
    # 早期混响用作训练时的label
    rir_early = rir[:, 0, :int(50 * 16000 / 1000)]
    speech_early = add_reverb(speech, rir_early)[0]  # (T,)

    speech_reverb = add_reverb(speech, rir[:, 0, :])  # (2, T)
    noise_reverb = add_reverb(noise, rir[:, 1, :])  # (2, T)

    amp = 0.5 * np.random.randn() + 0.5 + eps  # 随机幅度 让数据类型更多 amp (0.5, 1) + eps
    speech_early = amp * speech_early / (np.max(np.abs(speech_early)) + eps)

    # 归一化处理
    speech_reverb_norm = amp * speech_reverb / (np.max(np.abs(speech_reverb)) + eps)  # (2, T)
    noise_reverb_norm = amp * noise_reverb / (np.max(np.abs(noise_reverb)) + eps)  # (2, T)

    # 加噪声
    alpha = np.sqrt(np.var(speech_reverb_norm[0, :]) * (10 ** (-snr / 10)) / (np.var(noise_reverb_norm[0, :] + eps)))
    noisy_reverb = speech_reverb_norm + alpha * noise_reverb_norm  # (2, T)

    # 防止截幅
    M = max(np.max(abs(noisy_reverb)), np.max(abs(speech_reverb_norm)), np.max(abs(alpha * noise_reverb_norm))) + eps
    if M > 1.0:
        noisy_reverb = noisy_reverb / M
        speech_early = speech_early / M

    return noisy_reverb, speech_early  # (2, T) & (T,)


class Train_Dataset(data.Dataset):
    def __init__(
            self,
            fs=16000,
            length_in_seconds=8,
            num_tot=60000,
            num_per_epoch=10000,
            random_sample=True,
            random_start=True,
    ):
        super(Train_Dataset, self).__init__()
        self.train_clean_database = sorted(librosa.util.find_files(TRAIN_CLEAN_PATH, ext='wav'))[:num_tot]
        self.train_noise_database = sorted(pd.read_csv(TRAIN_NOISE_CSV_PATH)['file_dir'].tolist())[:num_tot]
        self.rir_list = librosa.util.find_files(RIR_PATH, ext='npy')

        self.L = int(fs * length_in_seconds)
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_tot = num_tot
        self.num_per_epoch = num_per_epoch
        self.random_sample = random_sample
        self.random_start = random_start

    def __len__(self):
        return self.num_per_epoch

    def __getitem__(self, idx):
        # 训练的时候随机采样
        clean_list = random.sample(self.train_clean_database, self.num_per_epoch)
        noise_list = random.sample(self.train_noise_database, self.num_per_epoch)

        # 裁剪
        if self.random_start:
            begin_s = int(self.fs * random.uniform(0, 10 - self.length_in_seconds))
            clean, _ = sf.read(clean_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
            noise, _ = sf.read(noise_list[idx], dtype='float32', start=begin_s, stop=begin_s + self.L)
        else:
            clean, _ = sf.read(clean_list[idx], dtype='float32', start=0, stop=self.L)
            noise, _ = sf.read(noise_list[idx], dtype='float32', start=0, stop=self.L)

        # 加入混响和噪声
        rir_idx = random.randint(0, len(self.rir_list) - 1)
        rir = np.load(self.rir_list[rir_idx])
        max_index = np.min(np.argmax(np.abs(rir), axis=-1))
        # 截断rir(把声源到麦克风的那一段传播路径截掉)
        rir = rir[:, :, max_index:]
        snr = random.uniform(-5, 15)
        noisy, clean_early = mk_mixture(clean, noise, rir, snr, eps=1e-8)

        return noisy.astype('float32'), clean_early.astype('float32')  # (2, T) & (T,)


class Valid_Dataset(data.Dataset):
    def __init__(self):
        super(Valid_Dataset, self).__init__()
        self.noisy_1_dir_path = os.path.join(VALID_DATASET_PATH, 'noisy_reverb_1')
        self.noisy_2_dir_path = os.path.join(VALID_DATASET_PATH, 'noisy_reverb_2')
        self.speech_dir_path = os.path.join(VALID_DATASET_PATH, 'speech_early')

        self.noisy_1_list = sorted(librosa.util.find_files(self.noisy_1_dir_path, ext='wav'))
        self.noisy_2_list = sorted(librosa.util.find_files(self.noisy_2_dir_path, ext='wav'))
        self.speech_list = sorted(librosa.util.find_files(self.speech_dir_path, ext='wav'))

    def __len__(self):
        return len(self.noisy_1_list)

    def __getitem__(self, idx):
        noisy_1, _ = sf.read(self.noisy_1_list[idx], dtype='float32')  # (T,)
        noisy_2, _ = sf.read(self.noisy_2_list[idx], dtype='float32')  # (T,)
        speech_early, _ = sf.read(self.speech_list[idx], dtype='float32')  # (T,)

        noisy = np.stack([noisy_1, noisy_2], axis=0)  # (2, T)

        return noisy, speech_early  # (2, T) & (T,)


if __name__ == '__main__':
    train_dataset = Train_Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (noisy, clean_early) in enumerate(train_loader):
        print(i, noisy.shape, clean_early.shape)

    valid_dataset = Valid_Dataset()
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (noisy, speech_early) in enumerate(valid_loader):
        print(i, noisy.shape, speech_early.shape)













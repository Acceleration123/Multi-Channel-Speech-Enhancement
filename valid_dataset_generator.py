import numpy as np
import soundfile as sf
import librosa
import random
from scipy import signal
import os
import argparse
import pandas as pd
from tqdm import tqdm


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


class Dataset_Generator:
    def __init__(self, args):
        self.speech_dir = args.speech_dir
        self.noise_csv = args.noise_csv_path
        self.rir_dir = args.rir_dir
        self.save_dir = args.save_dir

    def generator(self):
        speech_list = sorted(librosa.util.find_files(self.speech_dir, ext='wav'))

        # 噪声随机抽取
        noise_list_tot = sorted(pd.read_csv(self.noise_csv)['file_dir'].tolist())
        noise_list = random.sample(noise_list_tot, len(speech_list))

        rir_list = sorted(librosa.util.find_files(self.rir_dir, ext='npy'))

        noisy_reverb_1_dir = os.path.join(self.save_dir, 'noisy_reverb_1')
        noisy_reverb_2_dir = os.path.join(self.save_dir, 'noisy_reverb_2')
        speech_early_dir = os.path.join(self.save_dir, 'speech_early')

        os.makedirs(noisy_reverb_1_dir, exist_ok=True)
        os.makedirs(noisy_reverb_2_dir, exist_ok=True)
        os.makedirs(speech_early_dir, exist_ok=True)

        file_length = 4
        for i in tqdm(range(len(speech_list))):
            speech, _ = sf.read(speech_list[i], dtype='float32')
            speech_length = len(speech)

            noise, _ = sf.read(noise_list[i], dtype='float32')
            noise = noise[:speech_length]

            rir = np.load(rir_list[i])
            snr = random.uniform(-5, 15)

            noisy_reverb, speech_early = mk_mixture(speech, noise, rir, snr)
            noisy_reverb, speech_early = noisy_reverb.astype(np.float32), speech_early.astype(np.float32)

            noisy_reverb_1 = noisy_reverb[0, :].astype(np.float32)
            noisy_reverb_2 = noisy_reverb[1, :].astype(np.float32)
            speech_early = speech_early.astype(np.float32)
            
            sf.write(os.path.join(noisy_reverb_1_dir, f"{i:0{file_length}d}.wav"), noisy_reverb_1, 16000)
            sf.write(os.path.join(noisy_reverb_2_dir, f"{i:0{file_length}d}.wav"), noisy_reverb_2, 16000)
            sf.write(os.path.join(speech_early_dir, f"{i:0{file_length}d}.wav"), speech_early, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-speech_dir', '--speech_dir', type=str,
                        default="D:\\Users\\14979\\Desktop\\DNS5_16k\\dev_clean")

    parser.add_argument('-noise_csv', '--noise_csv_path', type=str,
                        default="D:\\Users\\Database\\valid_noise_data_new.csv")

    parser.add_argument('-rir_dir', '--rir_dir', type=str,
                        default="D:\\Users\\Database\\valid_rir")

    parser.add_argument('-save_dir', '--save_dir', type=str,
                        default="D:\\Users\\14979\\Desktop\\TGRU_dataset")

    args = parser.parse_args()

    dataset_generator = Dataset_Generator(args)
    dataset_generator.generator()














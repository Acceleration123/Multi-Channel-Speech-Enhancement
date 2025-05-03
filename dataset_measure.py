import argparse
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pystoi import stoi
from pesq import pesq


def measure(noisy_path, clean_path):
    score = []
    noisy_list = sorted(librosa.util.find_files(noisy_path, ext='wav'))
    clean_list = sorted(librosa.util.find_files(clean_path, ext='wav'))

    for idx in tqdm(range(len(noisy_list))):
        noisy, fs = sf.read(noisy_list[idx], dtype='float32')
        clean, _ = sf.read(clean_list[idx], dtype='float32')

        score_new = {
            'stoi': stoi(clean, noisy, fs),
            'pseq': pesq(fs, clean, noisy, 'wb')
        }

        score.append(score_new)

    score_df = pd.DataFrame(score)
    score_df.to_csv('score.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-noisy', '--noisy_path', type=str)
    parser.add_argument('-clean', '--clean_path', type=str)
    args = parser.parse_args()

    measure(args.noisy_path, args.clean_path)



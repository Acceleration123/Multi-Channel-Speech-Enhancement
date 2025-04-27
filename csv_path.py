import pandas as pd
from tqdm import tqdm
import argparse


def path_converter(old_path_csv, new_path):
    old_path_df = pd.read_csv(old_path_csv)
    new_path_df = []

    for i in tqdm(range(len(old_path_df))):
        old_path_linux = old_path_df['file_dir'][i]
        old_path_windows = old_path_linux.replace('/', '\\')
        new_path_windows = old_path_windows.replace('\\data\\ssd1', new_path)
        new_path_df.append({
            'file_dir': new_path_windows
        })

    new_path_df = pd.DataFrame(new_path_df)
    new_path_df.to_csv(old_path_csv.replace('.csv', '_new.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-old', '--old_path_csv', type=str)
    parser.add_argument('-new', '--new_path', type=str)
    args = parser.parse_args()

    path_converter(args.old_path_csv, args.new_path)


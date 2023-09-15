import os, glob
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def build_file_list(data_path, data_type):
    if data_type == 'LRS2':
        files = sorted(glob.glob(os.path.join(data_path, 'main', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'pretrain', '*', '*.mp4')))
    elif data_type == 'LRS3':
        files = sorted(glob.glob(os.path.join(data_path, 'trainval', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'pretrain', '*', '*.mp4')))
        files.extend(glob.glob(os.path.join(data_path, 'test', '*', '*.mp4')))
    else:
        raise NotImplementedError
    return [f.replace(data_path + '/', '')[:-4] for f in files]

def per_file(f, args):
    save_path = os.path.join(args.save_path, f)
    if os.path.exists(save_path + '.wav'): return
    if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vid_name = os.path.join(args.data_path, f + '.mp4')
    os.system(
        f'ffmpeg -loglevel panic -nostdin -y -i {vid_name} -acodec pcm_s16le -ar 16000 -ac 1 {save_path}.wav')

def main():
    parser = get_parser()
    args = parser.parse_args()
    file_lists = build_file_list(args.data_path, args.data_type)
    Parallel(n_jobs=3)(delayed(per_file)(f, args) for f in tqdm(file_lists))

def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="path for original data"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path for saving"
    )
    parser.add_argument(
        "--data_type", type=str, required=True, help="LRS2 or LRS3"
    )
    return parser


if __name__ == "__main__":
    main()
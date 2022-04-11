import os, glob, subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Grid_dir', type=str, default="Data dir to GRID_corpus")
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_imgs_aud')
    args = parser.parse_args()
    return args

args = parse_args()

vid_files = sorted(glob.glob(os.path.join(args.Grid_dir, '*', 'video', '*.mpg')))   #suppose the directory: Data_dir/subject/video/mpg files
for k, v in enumerate(vid_files):
    t, f_name = os.path.split(v)
    t, _ = os.path.split(t)
    _, sub_name = os.path.split(t)
    out_im = os.path.join(args.Output_dir, sub_name, 'video', f_name[:-4])
    if len(glob.glob(os.path.join(out_im, '*.png'))) < 75:  # Can resume after killed
        if not os.path.exists(out_im):
            os.makedirs(out_im)
        out_aud = os.path.join(args.Output_dir, sub_name, 'audio')
        if not os.path.exists(out_aud):
            os.makedirs(out_aud)
        subprocess.call(f'ffmpeg -y -i {v} -qscale:v 2 -r 25 {out_im}/%02d.png', shell=True)
        subprocess.call(f'ffmpeg -y -i {v} -ac 1 -acodec pcm_s16le -ar 16000 {os.path.join(out_aud, f_name[:-4] + ".wav")}', shell=True)
    print(f'{k}/{len(vid_files)}')

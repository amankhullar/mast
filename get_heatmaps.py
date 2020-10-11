import argparse
import os
import sys

import numpy as np
import seaborn as sns
sns.set()

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--att_wts_pth", type=str, default="./att_wts",
        help="Path to the stored attention weights main dir")
    args = parser.parse_args()

    fontsize=18

    vids = os.listdir(args.att_wts_pth)
    vids = [v for v in vids if os.path.isdir(os.path.join(args.att_wts_pth, v))]
    for vid in tqdm(vids):
        vid_pth = os.path.join(args.att_wts_pth, vid)
        figs = ['h3_att.npy', 'h2_att_img_txt.npy', 'h2_att_aud_txt.npy', 'h1_att_txt.npy', 'h1_att_img.npy', 'h1_att_aud.npy']
        for fig in figs:
            with open(os.path.join(vid_pth, fig), 'rb') as f:
                att_fig = np.load(f)
                nans = np.isnan(att_fig)        # NOTE : Check why nan values are being generated
                att_fig[nans] = 0
                # Remove averaging done on text while creating npy files
                if fig == 'h1_att_txt.npy':
                    att_fig *= 2

            if fig  == 'h3_att.npy':
                ax = sns.heatmap(att_fig)
                yticklabels = ['Video-Text', 'Audio-Text']
                ax.set_yticklabels(yticklabels, fontsize=fontsize)
                # ax.tick_params(axis='x', which='major', labelsize=10)
                ax.set_title('Attention between Audio-Text and Video-Text', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
            elif fig == 'h2_att_img_txt.npy':
                ax = sns.heatmap(att_fig)
                yticklabels = ['Text', 'Video']
                ax.set_yticklabels(yticklabels, fontsize=fontsize)
                ax.set_title('Attention between Text and Video', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
            elif fig == 'h2_att_aud_txt.npy':
                ax = sns.heatmap(att_fig, yticklabels=['Text', 'Audio'])
                yticklabels=['Text', 'Audio']
                ax.set_yticklabels(yticklabels, fontsize=fontsize)
                ax.set_title('Attention between Text and Audio', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
            else:
                ax = sns.heatmap(att_fig)
            if fig == 'h1_att_txt.npy':
                ax.set_title('Attention within text', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
                ax.set_ylabel('Text encoder timesteps', fontsize=fontsize)
            elif fig == 'h1_att_img.npy':
                ax.set_title('Attention within video', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
                ax.set_ylabel('Video encoder timesteps', fontsize=fontsize)
            elif fig == 'h1_att_aud.npy':
                ax.set_title('Attention within audio', fontsize=fontsize)
                ax.set_xlabel('Decoder timesteps', fontsize=fontsize)
                ax.set_ylabel('Audio encoder timesteps', fontsize=fontsize)

            fig1 = ax.get_figure()
            fig1.savefig(os.path.join(vid_pth, fig.replace('.npy','.png')))
            ax.get_figure().clf()


if __name__ == "__main__":
    """
    Script to convert the attention distributions in form of numpy arrays into visualizations.
    """
    main()

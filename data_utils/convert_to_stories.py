'''
Script to convert How2 data into CNN-DM type data format.
'''

import os
import sys

import numpy as np

def main(src_tran_pth, src_desc_pth, dest_pth):
    try:
        with open(src_tran_pth, 'r') as f:
            #print("Reading audio files")
            tran_files = f.readlines()
            #print("Length of audio files", len(aud_files))
    except Exception as e:
        print("Error in loading source transcript file with error : " + str(e))
        sys.exit()

    try:
        with open(src_desc_pth, 'r') as f:
            desc_files = f.readlines()
    except Exception as e:
        print("Error in loading source transcript file with error : " + str(e))
        sys.exit()

    assert len(tran_files) == len(desc_files), "Unequal length of transcription and description"
    if not os.path.exists(dest_pth):
        os.system('mkdir -p ' + dest_pth)

    for idx in range(len(tran_files)):
        tran_txt = tran_files[idx].split(' ')
        desc_txt = desc_files[idx].split(' ')
        assert tran_txt[0] == desc_txt[0], "Files with different names"
        file_name = tran_txt[0] + '.story'
        tran_txt = ' '.join(tran_txt[1:])
        desc_txt = ' '.join(desc_txt[1:])
        out_txt = tran_txt + '\n' + '@highlight' + '\n\n' + desc_txt

        try:
            with open(os.path.join(dest_pth, file_name), 'w') as f:
                f.write(out_txt)
        except Exception as e:
            print("Error in writing the out file with error : " + str(e))
            sys.exit()

if __name__ == "__main__":
    text_path = '/home/aman_khullar/PreSumm/raw_data'
    how2_dir_path = '/home/aman_khullar/how2'
    splits = ['devtest', 'cv', 'train']
    for split_name in splits:
        src_tran_path = os.path.join(how2_dir_path, 'text_300/sum_{}_300/tran.tok.txt'.format(split_name))
        src_desc_path = os.path.join(how2_dir_path, 'text_300/sum_{}_300/desc.tok.txt'.format(split_name))
        dest_pth = os.path.join(text_path, 'how2_stories/sum_{}_300'.format(split_name))
        main(src_tran_path, src_desc_path, dest_pth)

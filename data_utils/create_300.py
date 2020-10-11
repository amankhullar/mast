import os
import sys

import numpy as np

def main(src_path, mid_path, tgt_path, is_vid):
    try:
        with open(src_path, 'r') as f:
            #print("Reading audio files")
            aud_files = f.readlines()
            #print("Length of audio files", len(aud_files))
    except Excpetion as e:
        print("Error in loading source audio file with error : " + str(e))
        sys.exit()
    if is_vid:
        for aud in aud_files:
            aud = aud.replace('\n', '.npy')
            os.system('cp ' + mid_path + aud + ' ' + tgt_path)
    else:
        try:
            with open(mid_path, 'r') as f:
                #print("Reading the source transcripts")
                txt_files = f.readlines()
                #print("Length of text files", len(txt_files))
        except Exception as e:
            print("Error in loading transcript file with error : " + str(e))
            sys.exit()
        try:
            with open(tgt_path, 'a') as f:
                for txt in txt_files:
                    txt_file = txt.split(' ')[0] + '\n'          # The \n has been added in accordance with the audio input files
                    if txt_file in aud_files:
                        f.write(txt)
        except Exception as e:
            print("Error in loading targe transcript file with error : " + str(e))
            sys.exit()

if __name__ == "__main__":
    src_path = "/home/aman_khullar/how2/vid_id.txt"
    is_vid = False
    if is_vid:
        mid_path = "/home/aman_khullar/how2/video_action_features/"
        tgt_path = "/home/aman_khullar/how2/video_features_300/"
    else:
        mid_path = "/home/aman_khullar/how2/text/sum_devtest/desc.tok.txt"
        tgt_path = "/home/aman_khullar/how2/text_300/sum_devtest_300/desc.tok.txt"
    main(src_path, mid_path, tgt_path, is_vid)


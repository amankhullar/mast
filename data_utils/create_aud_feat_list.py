#/home/aman_khullar/how2/video_features_300/D3Vsicg-eTI.npy

import os

def main(src_path, dest_path, text_path):
    file_path = 'fbank_pitch_181506/concat'
    try:
        with open(src_path, 'r') as f:
            txt = f.readlines()
    except Exception as e:
        print("Could not open")
    print("Reached here")
    with open(dest_path, 'a') as f1:
        for i in txt:
            filename = i.split(' ')[0] + '.npy' + '\n'
            f1.write(os.path.join(text_path, file_path, filename))

if __name__ == "__main__":
    text_path = '/home/aman_khullar/how2/'
    splits = ['devtest', 'cv', 'train']
    for split_name in splits:
        src_path = os.path.join(text_path, 'text_300/sum_{}_300/tran.tok.txt'.format(split_name))
        dest_path = os.path.join(text_path, 'audio_{}_300.txt'.format(split_name))
        main(src_path, dest_path, text_path)

import os
import sys

def main(tran_pth, att_wts_pth):
    with open(tran_pth, 'r') as f:
        transcripts = f.readlines()
    transcripts = [t.strip() for t in transcripts]
    for trans in transcripts:
        temp = trans.split(' ')
        filename = temp[0]
        trans = ' '.join(temp[1:])
        with open(os.path.join(att_wts_pth, filename, 'tran.txt'), 'w') as f:
            f.write(trans)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tran_pth = os.path.join(base_dir, 'tran.tok.txt')
    att_wts_pth = os.path.join(base_dir, 'att_wts')
    main(tran_pth, att_wts_pth)

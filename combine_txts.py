'''
Script to collect all the predictions and the reference summaries from
different folders in att_vis and store them in two files
'''
import argparse
import os
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--att_wts_pth', type=str, default='att_wts',
        help='Path to directory containing the predictions and references')
    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    att_dir_pth = os.path.join(base_dir, args.att_wts_pth)

    # if os.path.exists(os.path.join(base_dir, 'all_hyps.txt')):
    #     os.system('rm ' + os.path.join(base_dir, 'all_hyps.txt'))
    # if os.path.exists(os.path.join(base_dir, 'all_refs.txt')):
    #     os.system('rm ' + os.path.join(base_dir, 'all_refs.txt'))

    fh = open(os.path.join(base_dir, 'all_hyps.txt'), 'w')
    fr = open(os.path.join(base_dir, 'all_refs.txt'), 'w')

    dirs = os.listdir(att_dir_pth)
    video_ids = []
    for dir in dirs:
        try:
            with open(os.path.join(att_dir_pth, dir, 'hyp.txt'), 'r') as f:
                hyp = f.read().strip() + '\n'
            with open(os.path.join(att_dir_pth, dir, 'ref.txt'), 'r') as f:
                ref = f.read().strip() + '\n'
            fh.write(hyp)
            fr.write(ref)
            video_ids.append(dir)
        except:
            pass

    print("all_hyps.txt and all_ref.txt have been generated.")
    # print("Video_ids: {}".format(video_ids))

if __name__ == "__main__":
    main()

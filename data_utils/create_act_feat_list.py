import os

def main(src_path, dest_path, text_path):
    file_path = 'video_features_300'
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
    src_path = os.path.join(text_path, 'text_300/sum_devtest_300/tran.tok.txt')
    dest_path = os.path.join(text_path, 'actions_devtest_300.txt')
    main(src_path, dest_path, text_path)

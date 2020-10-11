import os
import sys
import numpy as np

import torch
from transformers import *

MODELS = [(XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),]

for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text

    # Files to encode
    src_files = ["/home/aman_khullar/how2/text_300/sum_train_300/tran.tok.txt",
                 "/home/aman_khullar/how2/text_300/sum_cv_300/tran.tok.txt",
                 "/home/aman_khullar/how2/text_300/sum_devtest_300/tran.tok.txt",]
    tgt_path = "/home/aman_khullar/how2/text_xlnet_300"
    for idx, src_file in enumerate(src_files):
        print("Converting Folder : ", idx)
        try:
            with open(src_file, 'r') as f:
                txts = f.readlines()
        except Exception as e:
            print("Error in opening file with error : " + str(e))
            sys.exit()
        for cnt, txt in enumerate(txts):
            print("Converting file : ", cnt)
            txt = txt.split(' ')
            fname = txt[0] + '.npy'
            txt = ' '.join(txt[1:])
            input_ids = torch.tensor([tokenizer.encode(txt, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
            embed = last_hidden_states.squeeze(0)
            embed = embed.numpy()
            np.save(os.path.join(tgt_path, fname), embed)
    print("Completed generating Embedding")
            

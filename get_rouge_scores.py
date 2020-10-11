import json
import os
import pandas as pd

BASE_DIR = 'att_wts'
scores = {}

for vid in os.listdir(BASE_DIR):
    if 'DS_Store' in vid or '.txt' in vid:
        continue
    ref_path = os.path.join(BASE_DIR, vid, 'ref.txt')
    hyp_path = os.path.join(BASE_DIR, vid, 'hyp.txt')
    out_path = os.path.join(BASE_DIR, vid, 'rouge.csv')

    cmd = f"python -m rouge.rouge --target_filepattern={ref_path} --prediction_filepattern={hyp_path} --output_filename={out_path} --use_stemmer=true"

    print(f"Processing video {vid}")
    os.system(cmd)

    data = pd.read_csv(out_path)
    scores[vid] = data[data['score_type'] == 'rougeL-F']['mid'].values[0]

    print("-----------\n")

with open('rouge_scores.json', 'w') as f:
    json.dump(scores, f)

"""
Filters the reference and hypothesis summaries to remove frequently occuring words in order to calcualte Content-F1 score
"""

transcript_words = set(['the', 'to', 'and', 'you', 'a', 'it', 'that', 'of', 'is', 'i', 'going', 'we', 'in', 'your', 'this', "'s", 'so', 'on'])

summary_words = set(['in', 'a', 'this', 'to', 'free', 'the', 'video', 'and', 'learn', 'from', 'on', 'with', 'how', 'tips', 'for', 'of', 'expert', 'an'])

def process_file(read_path, save_path, stopwords):
    filtered_lines = []
    with open(read_path) as f, open(save_path, 'w') as fw:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            filtered_line = ' '.join([word for word in words if word not in stopwords])
            fw.write(filtered_line + '\n')
            filtered_lines.append(filtered_line)
    return filtered_lines

process_file('all_refs.txt', 'all_refs_filtered.txt', transcript_words)

process_file('all_hyps.txt', 'all_hyps_filtered.txt', summary_words)

# process_file('test_abs_bert_cnnd_res.100000.candidate', 'test_abs_bert_cnnd_res.100000.filtered.candidate', summary_words)
# process_file('test_abs_bert_cnnd_res.100000.gold', 'test_abs_bert_cnnd_res.100000.filtered.gold', summary_words)

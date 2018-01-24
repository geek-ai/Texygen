import os

from utils.metrics.collapse import collapse

root = 'data_coco/'

# origin = root + 'image_coco.txt'
all_file = os.listdir(root)

with open('collpse_for_' + 'image_coco', 'w') as log:
    for ngram in range(2, 6):
        for test_file in all_file:
            test_text = root+test_file
            bleu = collapse(test_text=test_text, gram=ngram)
            bleu.set_name('gram='+str(ngram)+'file'+test_file)
            score = bleu.get_score(is_fast=False, ignore=False)
            score = str(score)
            print(bleu.get_name() + score)
            log.write(bleu.get_name() + score + '\n')

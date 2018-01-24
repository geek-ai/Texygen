import os

from utils.metrics.Bleu import Bleu

root = 'data_coco/'

origin = root + 'image_coco.txt'
all_file = os.listdir(root)

with open('bleu for' + 'image_coco', 'w') as log:
    for ngram in range(2,6):
        for test_file in all_file:
            test_text = root+test_file
            bleu = Bleu(test_text=test_text, real_text=origin, gram=ngram)
            bleu.set_name('gram='+str(ngram)+'file'+test_file)
            score = bleu.get_score(is_fast=False, ignore=False)
            score = str(score)
            print(bleu.get_name() + score)
            log.write(bleu.get_name() + score)

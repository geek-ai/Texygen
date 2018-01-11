import json

from nltk import CFG
from nltk.parse.generate import generate

from utils.text_process import *


# demo_grammar = """
#   S -> NP VP
#   NP -> Det N
#   PP -> P NP
#   VP -> 'slept' | 'saw' NP | 'walked' PP
#   Det -> 'the' | 'a'
#   N -> 'man' | 'park' | 'dog'
#   P -> 'in' | 'with'
# """

# cfg_grammar = """
#   S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
#   PLUS -> '+'
#   SUB -> '-'
#   PROD -> '*'
#   DIV -> '/'
#   x -> 'x'
#
# """
# grammar = CFG.fromstring(cfg_grammar)
# print(grammar)
# sens = generate(grammar, depth=11)
# for sentence in generate(grammar, depth=11):
#     print(' '.join(sentence))



class OracleCfg:
    def __init__(self, cfg_grammar=None, origin_file='save/origin.txt', oracle_file='save/oracle.txt',
                 wi_dict='save/word_index_dict.json', iw_dict='save/index_word_dict.json',
                 sequence_length=None):
        if cfg_grammar is None:
            cfg_grammar = """
              S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
              PLUS -> '+'
              SUB -> '-'
              PROD -> '*'
              DIV -> '/'
              x -> 'x'
            """

        self.grammar = CFG.fromstring(cfg_grammar)
        self.origin_file = origin_file
        self.oracle_file = oracle_file
        self.wi_dict = wi_dict
        self.iw_dict = iw_dict
        self.sequence_length = sequence_length
        self.vocab_size = None
        return

    def generate_sentence(self, depth=9, num=30000):
        if num > 30000:
            num = 30000
        sentences = generate(self.grammar, depth=depth, n=num)
        with open(self.origin_file, 'w') as file:
            for s in sentences:
                file.write(' '.join(s) + '\n')

    def pre_process(self):
        tokens = get_tokenlized(self.origin_file)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.wi_dict, 'w') as outfile:
            json.dump(word_index_dict, outfile)
        with open(self.iw_dict, 'w') as outfile:
            json.dump(index_word_dict, outfile)
        if self.sequence_length is None:
            self.sequence_length = len(max(tokens, key=len))
        else:
            self.sequence_length = max(self.sequence_length, len(max(tokens, key=len)))
        self.vocab_size = len(word_index_dict)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return

    def generate_oracle(self):
        self.generate_sentence()
        self.pre_process()
